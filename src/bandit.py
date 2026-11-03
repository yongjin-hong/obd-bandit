"""
OBD 밴딧 시뮬레이션 부품 모음

① 피처 구성: 사용자 피처 인코딩, 아이템 피처 행렬, 후보 행렬 생성
② 정책: RandomPolicy, BernoulliTS, LinTS (공통 사용법: select / update)
③ 시뮬레이터: ReplayData, run_replay, run_repeated, summarize_results
"""
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import polars as pl


# =====================================================================
# ① 피처 구성
# =====================================================================

# ===== 컬럼 이름 상수 =====
USER_FEATURE_COLS = [f"user_feature_{feature_idx}" for feature_idx in range(4)]
ITEM_NUMERIC_COL = "item_feature_0"
ITEM_CATEGORICAL_COLS = ["item_feature_1", "item_feature_2", "item_feature_3"]
OTHER_LABEL = "__other__"


class FeatureBuilder:
    """
    LinTS 입력(82차원) 생성기

    입력 구성 (후보 아이템 1개 기준):
        [ 사용자 피처 원-핫 (26) | 아이템 피처 (28) | 상호작용 여부 × 아이템 피처 (28) ]

    - 사용자 피처: 노출 비중이 rare_share_threshold 미만인 범주는 피처별 "기타"로 묶음
    - 아이템 피처: item_feature_0은 수치 그대로, item_feature_1~3은 원-핫
    - 상호작용 대상: interaction_feature_col에서 노출 수 기준 interaction_category_rank번째 범주
      (rank는 0부터 시작. 기본값 user_feature_0의 1번째 = EDA의 uf0_1)
    """

    def __init__(
        self,
        interaction_feature_col: str = "user_feature_0",
        interaction_category_rank: int = 1,
        rare_share_threshold: float = 0.01,
    ):
        self.interaction_feature_col = interaction_feature_col
        self.interaction_category_rank = interaction_category_rank
        self.rare_share_threshold = rare_share_threshold

    # ----- 학습: 인코딩 기준 생성 -----
    def fit(self, log_df: pl.DataFrame, item_df: pl.DataFrame) -> "FeatureBuilder":
        self._fit_user_encoder(log_df)
        self._fit_item_matrix(item_df)
        return self

    def _fit_user_encoder(self, log_df: pl.DataFrame) -> None:
        # ===== 피처별 범주 목록 생성 (노출 많은 순, 희귀 범주 → 기타) =====
        self.user_categories: Dict[str, List[str]] = {}
        self.all_user_categories: Dict[str, List[str]] = {}
        total_rows = log_df.height

        for feature_col in USER_FEATURE_COLS:
            category_count_df = (
                log_df[feature_col]
                .value_counts()
                .sort(["count", feature_col], descending=[True, False])
            )
            all_categories = category_count_df[feature_col].to_list()
            kept_categories = (
                category_count_df
                .filter(pl.col("count") / total_rows >= self.rare_share_threshold)[feature_col]
                .to_list()
            )
            self.all_user_categories[feature_col] = all_categories
            self.user_categories[feature_col] = kept_categories + [OTHER_LABEL]

        # ===== 피처별 원-핫 시작 위치(offset) 계산 =====
        self.user_offsets: Dict[str, int] = {}
        current_offset = 0
        for feature_col in USER_FEATURE_COLS:
            self.user_offsets[feature_col] = current_offset
            current_offset += len(self.user_categories[feature_col])
        self.n_user_dims = current_offset

        # ===== 상호작용 대상 범주 확정 (노출 수 순위 기준) =====
        self.interaction_category = self.all_user_categories[self.interaction_feature_col][
            self.interaction_category_rank
        ]

    def _fit_item_matrix(self, item_df: pl.DataFrame) -> None:
        # ===== item_id 순서 정렬 및 연속성 확인 (0 ~ n_items-1) =====
        sorted_item_df = item_df.sort("item_id")
        item_ids = sorted_item_df["item_id"].to_list()
        if item_ids != list(range(len(item_ids))):
            raise ValueError("item_id가 0부터 연속된 정수가 아닙니다.")
        self.n_items = len(item_ids)

        # ===== 수치형 피처: 값 그대로 사용 =====
        matrix_blocks = [sorted_item_df[ITEM_NUMERIC_COL].to_numpy().reshape(-1, 1).astype(float)]
        self.item_feature_names = [ITEM_NUMERIC_COL]

        # ===== 범주형 피처: 원-핫 (범주는 문자열 정렬 순) =====
        self.item_categories: Dict[str, List[str]] = {}
        for feature_col in ITEM_CATEGORICAL_COLS:
            categories = sorted(sorted_item_df[feature_col].unique().to_list())
            self.item_categories[feature_col] = categories
            category_index = {category: idx for idx, category in enumerate(categories)}

            onehot_block = np.zeros((self.n_items, len(categories)))
            for row_idx, category in enumerate(sorted_item_df[feature_col].to_list()):
                onehot_block[row_idx, category_index[category]] = 1.0
            matrix_blocks.append(onehot_block)
            self.item_feature_names += [f"{feature_col}={category[:8]}" for category in categories]

        self.item_matrix = np.hstack(matrix_blocks)  # (n_items, n_item_dims)
        self.n_item_dims = self.item_matrix.shape[1]

    # ----- 변환: 로그 → 사용자 피처 행렬, 상호작용 여부 -----
    def transform_users(self, log_df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            user_matrix: (n_rows, n_user_dims) 사용자 피처 원-핫
            interaction_flags: (n_rows,) 상호작용 대상 범주이면 1.0, 아니면 0.0
        """
        n_rows = log_df.height
        user_matrix = np.zeros((n_rows, self.n_user_dims))
        row_positions = np.arange(n_rows)

        # ===== 피처별 범주 → 원-핫 위치 변환 (학습 때 없던 범주, 희귀 범주 → 기타) =====
        for feature_col in USER_FEATURE_COLS:
            categories = self.user_categories[feature_col]
            other_index = len(categories) - 1
            category_index = {category: idx for idx, category in enumerate(categories[:-1])}

            local_indices = (
                log_df[feature_col]
                .replace_strict(category_index, default=other_index, return_dtype=pl.Int64)
                .to_numpy()
            )
            user_matrix[row_positions, self.user_offsets[feature_col] + local_indices] = 1.0

        # ===== 상호작용 대상 여부 =====
        interaction_flags = (
            (log_df[self.interaction_feature_col] == self.interaction_category)
            .cast(pl.Float64)
            .to_numpy()
        )
        return user_matrix, interaction_flags

    # ----- 후보 행렬: 사용자 1명 × 후보 아이템 전체 -----
    def build_candidates(self, user_vector: np.ndarray, interaction_flag: float) -> np.ndarray:
        """
        Returns:
            candidate_matrix: (n_items, n_dims). i번째 행 = 아이템 i를 추천할 때의 입력 벡터
        """
        user_block = np.tile(user_vector, (self.n_items, 1))
        interaction_block = interaction_flag * self.item_matrix
        return np.hstack([user_block, self.item_matrix, interaction_block])

    # ----- 확인용 정보 -----
    @property
    def n_dims(self) -> int:
        return self.n_user_dims + 2 * self.n_item_dims

    @property
    def user_feature_names(self) -> List[str]:
        names = []
        for feature_col in USER_FEATURE_COLS:
            short_prefix = feature_col.replace("user_feature_", "uf")
            for category in self.user_categories[feature_col]:
                label = "other" if category == OTHER_LABEL else category[:8]
                names.append(f"{short_prefix}={label}")
        return names


# =====================================================================
# ② 정책
# =====================================================================
#
# 공통 사용법
#   - select(candidate_matrix) -> 선택한 아이템 번호
#   - update(candidate_matrix, item, reward) -> 선택한 아이템의 결과로 학습
#   - uses_context: True면 시뮬레이터가 후보 행렬을 만들어 넘김, False면 None을 넘김
# =====================================================================

# ===== 공통: 최대값이 여러 개면 그중 무작위 선택 (피처가 같은 아이템 쌍 대응) =====
def argmax_random_tie(scores: np.ndarray, rng: np.random.Generator) -> int:
    best_items = np.flatnonzero(scores == scores.max())
    if best_items.size == 1:  # 동점이 없으면 난수 생성 생략 (속도)
        return int(best_items[0])
    return int(rng.choice(best_items))


class BasePolicy:
    uses_context = False

    def __init__(self, n_items: int, seed: Optional[int] = None):
        self.n_items = n_items
        self.rng = np.random.default_rng(seed)

    def select(self, candidate_matrix: Optional[np.ndarray]) -> int:
        raise NotImplementedError

    def update(self, candidate_matrix: Optional[np.ndarray], item: int, reward: float) -> None:
        raise NotImplementedError


class RandomPolicy(BasePolicy):
    """모든 아이템을 균등 확률로 선택 (학습 없음, 시뮬레이터 검증용)"""

    def select(self, candidate_matrix: Optional[np.ndarray]) -> int:
        return int(self.rng.integers(self.n_items))

    def update(self, candidate_matrix: Optional[np.ndarray], item: int, reward: float) -> None:
        pass


class BernoulliTS(BasePolicy):
    """
    비컨텍스추얼 Thompson Sampling
    - 아이템마다 클릭률을 Beta(alpha, beta) 분포로 표현
    - 매 라운드: 아이템별 분포에서 값 하나씩 샘플링 → 가장 큰 아이템 선택
    - 학습: 클릭이면 alpha += 1, 아니면 beta += 1
    """

    def __init__(self, n_items: int, prior_alpha: float = 1.0, prior_beta: float = 1.0,
                 seed: Optional[int] = None):
        super().__init__(n_items, seed)
        self.alpha = np.full(n_items, prior_alpha)
        self.beta = np.full(n_items, prior_beta)

    def select(self, candidate_matrix: Optional[np.ndarray]) -> int:
        sampled_ctrs = self.rng.beta(self.alpha, self.beta)
        return argmax_random_tie(sampled_ctrs, self.rng)

    def update(self, candidate_matrix: Optional[np.ndarray], item: int, reward: float) -> None:
        self.alpha[item] += reward
        self.beta[item] += 1.0 - reward


class LinTS(BasePolicy):
    """
    컨텍스추얼 Thompson Sampling (공유형 선형 모델)
    - 클릭 여부를 입력 벡터의 선형 결합으로 예측: reward ≈ x · theta
    - theta의 불확실성을 정규분포 N(mean, exploration_scale² × A⁻¹)로 표현
        A = regularization × I + Σ x xᵀ  (선택·관측된 입력들의 누적)
        b = Σ reward × x
        mean = A⁻¹ b
    - 매 라운드: theta를 분포에서 샘플링 → 후보 행렬 × theta로 점수 계산 → 최대 점수 아이템 선택
    - exploration_scale이 클수록 샘플링 폭이 넓어져 탐색을 많이 함
    """

    uses_context = True

    def __init__(self, n_items: int, n_dims: int, exploration_scale: float = 0.5,
                 regularization: float = 1.0, seed: Optional[int] = None):
        super().__init__(n_items, seed)
        self.n_dims = n_dims
        self.exploration_scale = exploration_scale

        # ===== 사후분포 초기화 (A = λI → A⁻¹ = I/λ, b = 0) =====
        self.precision_inverse = np.eye(n_dims) / regularization  # A⁻¹
        self.reward_weighted_sum = np.zeros(n_dims)                # b
        self._refresh_posterior()

    def _refresh_posterior(self) -> None:
        # ===== 평균과 샘플링용 촐레스키 분해 갱신 (학습할 때만 호출 → 매 라운드 계산 비용 절약) =====
        self.theta_mean = self.precision_inverse @ self.reward_weighted_sum
        covariance = (self.exploration_scale ** 2) * self.precision_inverse
        covariance = (covariance + covariance.T) / 2  # 수치 오차로 대칭이 깨지는 것 방지
        self.covariance_cholesky = np.linalg.cholesky(covariance)

    def select(self, candidate_matrix: Optional[np.ndarray]) -> int:
        # ===== theta 샘플링: mean + L z  (z ~ 표준정규) =====
        standard_normal = self.rng.standard_normal(self.n_dims)
        sampled_theta = self.theta_mean + self.covariance_cholesky @ standard_normal

        # ===== 후보별 점수 계산 후 최대값 선택 =====
        scores = candidate_matrix @ sampled_theta
        return argmax_random_tie(scores, self.rng)

    def update(self, candidate_matrix: Optional[np.ndarray], item: int, reward: float) -> None:
        chosen_vector = candidate_matrix[item]

        # ===== A⁻¹ 갱신 (Sherman-Morrison 공식: 역행렬을 다시 계산하지 않고 바로 갱신) =====
        inverse_times_vector = self.precision_inverse @ chosen_vector
        denominator = 1.0 + chosen_vector @ inverse_times_vector
        self.precision_inverse -= np.outer(inverse_times_vector, inverse_times_vector) / denominator

        # ===== b 갱신 및 사후분포 재계산 =====
        self.reward_weighted_sum += reward * chosen_vector
        self._refresh_posterior()


# =====================================================================
# ③ 시뮬레이터
# =====================================================================

class ReplayData:
    """replay에 필요한 로그 정보를 numpy 배열로 모아둔 묶음 (기간별로 하나씩 생성)"""

    def __init__(self, log_df: pl.DataFrame, feature_builder: FeatureBuilder):
        self.feature_builder = feature_builder
        self.logged_items = log_df["item_id"].to_numpy()
        self.logged_clicks = log_df["click"].to_numpy().astype(float)
        self.user_matrix, self.interaction_flags = feature_builder.transform_users(log_df)
        self.n_rows = log_df.height
        self.logged_ctr = float(self.logged_clicks.mean())


class ReplayResult:
    """replay 1회 실행 결과"""

    def __init__(self, matched_clicks: np.ndarray, n_rounds: int, elapsed_seconds: float):
        self.matched_clicks = matched_clicks      # 일치한 라운드들의 클릭 여부 (시간순)
        self.n_rounds = n_rounds                  # 전체 라운드 수
        self.n_matched = len(matched_clicks)      # 일치 라운드 수
        self.n_clicks = int(matched_clicks.sum())
        self.ctr = float(matched_clicks.mean()) if self.n_matched > 0 else float("nan")
        self.elapsed_seconds = elapsed_seconds

    @property
    def cumulative_ctr(self) -> np.ndarray:
        """일치 라운드 기준 누적 클릭률 추이"""
        return np.cumsum(self.matched_clicks) / np.arange(1, self.n_matched + 1)


def run_replay(policy: BasePolicy, replay_data: ReplayData, row_order: np.ndarray) -> ReplayResult:
    """
    replay 1회 실행
    - 로그를 row_order 순서로 한 줄씩 읽음
    - 정책이 고른 아이템 == 로그의 아이템 → 실제 클릭 결과를 기록하고 학습
    - 다르면 → 그 줄은 버림 (결과를 알 수 없으므로)
    """
    start_time = time.perf_counter()
    matched_clicks = []

    for row_idx in row_order:
        # ===== 정책에 넘길 컨텍스트 (컨텍스추얼 정책만 후보 행렬 생성) =====
        if policy.uses_context:
            context = replay_data.feature_builder.build_candidates(
                replay_data.user_matrix[row_idx], replay_data.interaction_flags[row_idx]
            )
        else:
            context = None

        # ===== 선택 → 로그와 일치하면 기록 및 학습 =====
        chosen_item = policy.select(context)
        if chosen_item == replay_data.logged_items[row_idx]:
            reward = replay_data.logged_clicks[row_idx]
            policy.update(context, chosen_item, reward)
            matched_clicks.append(reward)

    elapsed_seconds = time.perf_counter() - start_time
    return ReplayResult(np.array(matched_clicks), len(row_order), elapsed_seconds)


def run_repeated(
    policy_factory: Callable[[int], BasePolicy],
    replay_data: ReplayData,
    n_runs: int,
    base_seed: int = 0,
    verbose: bool = True,
) -> List[ReplayResult]:
    """
    replay 반복 실행
    - 매 실행마다 로그 순서를 무작위로 섞고, 정책을 새로 생성(처음부터 학습)
    - policy_factory: seed를 받아 새 정책을 만드는 함수 (예: lambda seed: BernoulliTS(34, seed=seed))
    """
    order_rng = np.random.default_rng(base_seed)
    results = []

    for run_idx in range(n_runs):
        # ===== 로그 순서 섞기 → 새 정책으로 replay 1회 =====
        row_order = order_rng.permutation(replay_data.n_rows)
        policy = policy_factory(base_seed + run_idx)
        result = run_replay(policy, replay_data, row_order)
        results.append(result)

        if verbose:
            print(f"  run {run_idx + 1:>2}/{n_runs}: 일치 {result.n_matched:,} / 클릭 {result.n_clicks} "
                  f"/ CTR {result.ctr:.4%} ({result.elapsed_seconds:.1f}초)")
    return results


def summarize_results(results: List[ReplayResult], label: str) -> Dict[str, float]:
    """반복 실행 결과 요약: CTR 평균과 95% 구간(2.5~97.5 백분위), 평균 일치 라운드 수"""
    ctr_values = np.array([result.ctr for result in results])
    return {
        "policy": label,
        "ctr_mean": float(ctr_values.mean()),
        "ctr_p2.5": float(np.percentile(ctr_values, 2.5)),
        "ctr_p97.5": float(np.percentile(ctr_values, 97.5)),
        "matched_mean": float(np.mean([result.n_matched for result in results])),
        "clicks_mean": float(np.mean([result.n_clicks for result in results])),
        "n_runs": len(results),
    }
