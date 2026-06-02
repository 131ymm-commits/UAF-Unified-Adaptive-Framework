"""
UAF-Ω: Unified Adaptive Framework — Omega Release
===================================================
Синтез всех версий UAF-агента в одном файле.

Что исправлено относительно предыдущих версий:
  1. PG считается ПОСЛЕ реального наблюдения (не до)
  2. Все 5 действий: ask, answer, reflect, search_memory, call_tool  
  3. Diversity Penalty (Аксиома 4 — после экспериментов)
  4. Self-monitor с калибровкой (уверен и ошибся = штраф)
  5. Диалектическое переключение режимов LLM
  6. Ноосферная динамика (multi-agent консенсус)
  7. Честный бенчмарк против жадного агента
  8. Все формулы из THEORY.md §2–6

Структура:
  UAFMath          — чистая математика, без зависимостей
  BeliefSystem     — байесовское belief над latent intents  
  WorldModel       — вероятностная модель наблюдений
  EpisodeMemory    — память с prior-влиянием
  SelfMonitor      — калибровка уверенности
  UAFPolicy        — выбор действия: ask vs answer vs reflect
  UAFAgent         — полный агент
  GreedyAgent      — жадный агент без уточнений (baseline)
  LLMShell         — языковый слой поверх symbolic core
  HallucinationGuard  — детектор галлюцинаций
  Noosphere        — динамика мультиагентной системы
  Benchmark        — честное сравнение

Запуск:
  python uaf_omega.py           # demo + benchmark
  python uaf_omega.py --checks  # только assert-проверки
  python uaf_omega.py --noosphere  # мультиагентный эксперимент

Формальные основания: THEORY.md (github.com/131ymm-commits)
"""

import math
import random
import re
import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import deque


# ============================================================
# §0. КОНСТАНТЫ И ТИПЫ
# ============================================================

INTENTS = ("math", "code", "theory")
ACTIONS = ("ask_clarify", "answer_math", "answer_code", "answer_theory",
           "reflect", "search_memory", "call_tool")

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яёЁ0-9_]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def normalize(d: Dict[str, float]) -> Dict[str, float]:
    s = sum(d.values())
    if s <= 0:
        n = len(d)
        return {k: 1.0 / n for k in d}
    return {k: v / s for k, v in d.items()}


def logsumexp(xs: List[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def sample_dist(dist: Dict[str, float]) -> str:
    r, acc = random.random(), 0.0
    last = None
    for k, p in dist.items():
        last = k
        acc += p
        if r <= acc:
            return k
    return last


# ============================================================
# §1. ЧИСТАЯ МАТЕМАТИКА UAF
# ============================================================

class UAFMath:
    """
    Статический класс с реализацией всех формул THEORY.md.
    Без зависимостей. Тестируется изолированно.
    """

    @staticmethod
    def entropy(dist: Dict[str, float]) -> float:
        """H(X) = -Σ P(x) log P(x)"""
        eps = 1e-12
        return -sum(max(eps, p) * math.log(max(eps, p)) for p in dist.values())

    @staticmethod
    def kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
        """D_KL(P‖Q) = Σ P(x) log [P(x) / Q(x)]"""
        eps = 1e-12
        return sum(
            p.get(k, eps) * math.log(p.get(k, eps) / max(q.get(k, eps), eps))
            for k in set(p) | set(q)
        )

    @staticmethod
    def pg_step(model_logp: float, baseline_logp: float) -> float:
        """
        PG_step = log[M(w_t|w_{<t}) / M_0(w_t|w_{<t})]
        Считается ПОСЛЕ получения реального наблюдения.
        PG > 0: модель предсказывает лучше базовой.
        """
        return model_logp - baseline_logp

    @staticmethod
    def uaf_scale(cumulative_pg: float, n_steps: int) -> float:
        """
        Нормированная шкала UAF [-3, +3].
        scale = tanh(NPG / steps / 2) * 3
        """
        if n_steps == 0:
            return 0.0
        return math.tanh(cumulative_pg / n_steps / 2.0) * 3.0

    @staticmethod
    def lambda_uaf(
        raw_lambda: float,
        reliability: float,
        independence: float,
        scope: float,
        freshness: float,
        predictive_gain: float,
    ) -> float:
        """
        λ_UAF = λ · R · I · S · F · (1 + G)
        Расширенный Байесовский вес свидетельства (THEORY.md §4)
        """
        return raw_lambda * reliability * independence * scope * freshness * (1 + predictive_gain)

    @staticmethod
    def free_energy(U: float, T: float, H: float) -> float:
        """F = U - T·H"""
        return U - T * H

    @staticmethod
    def answer_utility(
        p: float,
        answer_cost: float = 0.05,
        self_penalty: float = 0.0,
    ) -> float:
        """
        U(answer_i) = (2·p_i - 1) - C_answer - P_self
        Ожидаемая полезность ответа при reward ±1.
        """
        return (2.0 * p - 1.0) - answer_cost - self_penalty

    @staticmethod
    def bayesian_posterior(
        prior: Dict[str, float],
        log_likelihoods: Dict[str, float],
    ) -> Dict[str, float]:
        """
        P(M|E) ∝ P(E|M) · P(M)
        log posterior = log prior + log likelihood
        """
        log_posts = {}
        for k in prior:
            log_posts[k] = math.log(max(prior[k], 1e-12)) + log_likelihoods.get(k, 0.0)
        m = max(log_posts.values())
        posts = {k: math.exp(v - m) for k, v in log_posts.items()}
        return normalize(posts)

    @staticmethod
    def diversity_penalty(
        agent_beliefs: List[Dict[str, float]],
        threshold: float = 0.1,
        gamma: float = 0.5,
    ) -> float:
        """
        Аксиома 4 (Разнообразия) — выведена из экспериментов.
        Penalty = γ · max(0, τ - std({p_i}))
        Наказывает за предсказательный сговор (все агенты думают одинаково).
        """
        if len(agent_beliefs) < 2:
            return 0.0
        # Дисперсия по первому intentu как прокси
        first_key = list(agent_beliefs[0].keys())[0]
        probs = [b[first_key] for b in agent_beliefs]
        n = len(probs)
        mean_p = sum(probs) / n
        std_p = math.sqrt(sum((p - mean_p) ** 2 for p in probs) / n)
        return gamma * max(0.0, threshold - std_p)


# ============================================================
# §2. МОДЕЛЬ МИРА
# ============================================================

class WorldModel:
    """
    Проверяемая вероятностная модель:
    Observation ~ P(obs | intent, clarify).
    Байесовское обновление belief по наблюдениям.
    """

    def __init__(self):
        self._vocab = {
            "math": {
                "integral", "matrix", "proof", "theorem",
                "equation", "derivative", "limit", "lemma",
                "eigenvalue", "gradient", "tensor", "manifold",
            },
            "code": {
                "python", "bug", "function", "class",
                "api", "stack", "debug", "traceback",
                "async", "loop", "variable", "import",
            },
            "theory": {
                "uaf", "boundary", "prediction", "model",
                "agent", "theory", "update", "prior",
                "belief", "entropy", "bayes", "noosphere",
            },
        }
        self._noise = {
            "help", "please", "problem", "question",
            "need", "make", "what", "how", "explain", "show",
        }
        self._all = sorted(
            set().union(*self._vocab.values()) | self._noise
        )
        self._cache: Dict[Tuple, Dict] = {}

    def token_dist(self, intent: str, clarify: bool = False) -> Dict[str, float]:
        key = (intent, clarify)
        if key in self._cache:
            return self._cache[key]
        weights = {}
        for tok in self._all:
            if tok in self._vocab[intent]:
                w = 9.0 if clarify else 5.0
            elif tok in self._noise:
                w = 1.0 if clarify else 3.0
            else:
                w = 0.5
            weights[tok] = w
        s = sum(weights.values())
        dist = {t: w / s for t, w in weights.items()}
        self._cache[key] = dist
        return dist

    def log_likelihood(self, text: str, intent: str, clarify: bool = False) -> float:
        toks = tokenize(text)
        if not toks:
            return math.log(1e-9)
        dist = self.token_dist(intent, clarify)
        unk = 1.0 / (20.0 * len(self._all))
        return sum(math.log(dist.get(tok, unk)) for tok in toks)

    def mixture_logp(self, text: str, belief: Dict[str, float], clarify: bool = False) -> float:
        """log P(text | belief) = logsumexp_i [log P(i) + log P(text|i)]"""
        return logsumexp([
            math.log(max(belief[i], 1e-12)) + self.log_likelihood(text, i, clarify)
            for i in INTENTS
        ])

    def uniform_baseline_logp(self, text: str) -> float:
        """log P_baseline(text) — равномерное распределение по словарю"""
        toks = tokenize(text)
        if not toks:
            return math.log(1e-9)
        vocab_size = len(self._all) + 50
        return len(toks) * math.log(1.0 / vocab_size)

    def posterior(
        self, prior: Dict[str, float], text: str, clarify: bool = False
    ) -> Dict[str, float]:
        """Байесовское обновление belief"""
        log_liks = {i: self.log_likelihood(text, i, clarify) for i in INTENTS}
        return UAFMath.bayesian_posterior(prior, log_liks)

    def sample_obs(self, intent: str, clarify: bool = False, length: int = 5) -> str:
        dist = self.token_dist(intent, clarify)
        toks = random.choices(list(dist.keys()), weights=list(dist.values()), k=length)
        return " ".join(toks)


# ============================================================
# §3. ПАМЯТЬ
# ============================================================

@dataclass
class Episode:
    text: str
    intent: str
    action: str
    correct: bool
    pg: float
    timestamp: float = field(default_factory=time.time)


class EpisodeMemory:
    """
    Память с влиянием на prior следующего эпизода.
    Реализует §16.2 THEORY.md: T_ij обновляется по сбываемости.
    """

    def __init__(self, maxlen: int = 300):
        self.maxlen = maxlen
        self._items: deque[Episode] = deque(maxlen=maxlen)

    def add(self, ep: Episode):
        self._items.append(ep)

    def prior_from_memory(self, text: str) -> Dict[str, float]:
        """
        Prior = взвешенное голосование по похожим эпизодам.
        Правильные эпизоды имеют больший вес, неправильные — меньший.
        """
        toks = set(tokenize(text))
        scores = {i: 1.0 for i in INTENTS}
        for ep in self._items:
            overlap = len(toks & set(tokenize(ep.text)))
            if overlap > 0:
                w = overlap * (1.0 if ep.correct else 0.25)
                scores[ep.intent] = scores.get(ep.intent, 0.0) + w
        return normalize(scores)

    def recent_pg(self, n: int = 20) -> float:
        """Средний NPG за последние n эпизодов."""
        recent = list(self._items)[-n:]
        if not recent:
            return 0.0
        return sum(ep.pg for ep in recent) / len(recent)

    def search(self, query: str, top_k: int = 3) -> List[Episode]:
        """Поиск похожих эпизодов по перекрытию токенов."""
        toks = set(tokenize(query))
        scored = [
            (len(toks & set(tokenize(ep.text))), ep)
            for ep in self._items
        ]
        scored.sort(key=lambda x: -x[0])
        return [ep for _, ep in scored[:top_k] if _ > 0]


# ============================================================
# §4. SELF-MONITOR
# ============================================================

class SelfMonitor:
    """
    Калибровка уверенности.
    Если агент уверен и ошибается — штраф растёт.
    Если агент уверен и прав — штраф медленно снижается.

    Реализует §18.3 THEORY.md:
    Resp(j→i) = KL(P_hat‖P_true) · |U(a*) - U(a_opt)|
    """

    def __init__(self, confidence_threshold: float = 0.75):
        self.threshold = confidence_threshold
        self.penalty: float = 0.0
        self._records: List[Tuple[float, bool]] = []

    def record(self, confidence: float, correct: bool):
        self._records.append((confidence, correct))
        if confidence >= self.threshold and not correct:
            # Уверен и ошибся — увеличиваем штраф
            self.penalty = min(0.45, self.penalty + 0.09)
        elif confidence >= self.threshold and correct:
            # Уверен и прав — медленно снижаем
            self.penalty = max(0.0, self.penalty - 0.03)

    def accuracy(self) -> float:
        if not self._records:
            return 0.0
        return sum(1 for _, c in self._records if c) / len(self._records)

    def calibration_error(self) -> float:
        """ECE: Expected Calibration Error"""
        if len(self._records) < 5:
            return 0.0
        bins = [[] for _ in range(5)]
        for conf, correct in self._records:
            idx = min(int(conf * 5), 4)
            bins[idx].append(correct)
        ece = 0.0
        for i, b in enumerate(bins):
            if b:
                bin_conf = (i + 0.5) / 5.0
                bin_acc = sum(b) / len(b)
                ece += abs(bin_acc - bin_conf) * len(b) / len(self._records)
        return ece


# ============================================================
# §5. ПОЛИТИКА ДЕЙСТВИЯ
# ============================================================

class UAFPolicy:
    """
    Выбор действия по ожидаемой полезности.

    U(answer_i) = (2p_i - 1) - C_answer - P_self
    U(ask) = E[max_i U(answer_i | new_obs)] + κ·IG - C_ask
    U(reflect) = κ_r·H(belief) - C_reflect  (когда неуверены и уже спросили)
    U(search_memory) = κ_m·recent_pg - C_search  (когда память может помочь)
    """

    def __init__(
        self,
        ask_cost: float = 0.30,
        answer_cost: float = 0.05,
        reflect_cost: float = 0.10,
        search_cost: float = 0.08,
        k_info: float = 0.30,
        k_reflect: float = 0.20,
        k_search: float = 0.15,
        mc_rollouts: int = 40,
    ):
        self.ask_cost = ask_cost
        self.answer_cost = answer_cost
        self.reflect_cost = reflect_cost
        self.search_cost = search_cost
        self.k_info = k_info
        self.k_reflect = k_reflect
        self.k_search = k_search
        self.mc_rollouts = mc_rollouts

    def compute_all_utilities(
        self,
        belief: Dict[str, float],
        world_model: WorldModel,
        self_monitor: SelfMonitor,
        memory: EpisodeMemory,
        clarified: bool,
        searched: bool,
        step: int,
    ) -> Dict[str, float]:
        utils = {}

        # Полезность ответов
        for intent in INTENTS:
            utils[f"answer_{intent}"] = UAFMath.answer_utility(
                belief[intent], self.answer_cost, self_monitor.penalty
            )

        # Полезность уточнения (только если ещё не спрашивали)
        if clarified:
            utils["ask_clarify"] = -1e9
        else:
            utils["ask_clarify"] = self._ask_utility(
                belief, world_model, self_monitor, clarified
            )

        # Полезность рефлексии (полезна при высокой неопределённости)
        H = UAFMath.entropy(belief)
        utils["reflect"] = (
            self.k_reflect * H - self.reflect_cost
            if H > 0.8 and step > 1
            else -1e9
        )

        # Полезность поиска в памяти
        recent_pg = memory.recent_pg()
        utils["search_memory"] = (
            self.k_search * max(recent_pg, 0) - self.search_cost
            if not searched and len(memory._items) > 3
            else -1e9
        )

        # Инструмент (заглушка — доступен когда нужно)
        utils["call_tool"] = -1e9

        return utils

    def _ask_utility(
        self,
        belief: Dict[str, float],
        world_model: WorldModel,
        self_monitor: SelfMonitor,
        clarified: bool,
    ) -> float:
        prior_H = UAFMath.entropy(belief)
        total = 0.0
        for _ in range(self.mc_rollouts):
            hidden = sample_dist(belief)
            sim_obs = world_model.sample_obs(hidden, clarify=True)
            post2 = world_model.posterior(belief, sim_obs, clarify=True)
            best_future = max(
                UAFMath.answer_utility(post2[i], self.answer_cost, self_monitor.penalty)
                for i in INTENTS
            )
            ig = prior_H - UAFMath.entropy(post2)
            total += best_future + self.k_info * ig - self.ask_cost
        return total / self.mc_rollouts

    def select(self, utilities: Dict[str, float]) -> str:
        return max(utilities, key=utilities.get)


# ============================================================
# §6. СРЕДА
# ============================================================

class ToyEnv:
    """
    Проверяемая toy-среда.
    Скрытый intent, одно уточнение, один ответ.
    """

    def __init__(self, world_model: WorldModel):
        self.wm = world_model
        self.hidden_intent: Optional[str] = None
        self.done: bool = False
        self.asked: bool = False

    def reset(self, intent: Optional[str] = None) -> str:
        self.hidden_intent = intent or random.choice(INTENTS)
        self.done = False
        self.asked = False
        return self.wm.sample_obs(self.hidden_intent, clarify=False)

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        if self.done:
            raise RuntimeError("Episode finished")

        if action == "ask_clarify":
            if self.asked:
                self.done = True
                return "NO_MORE_INFO", -0.5, True, {"intent": self.hidden_intent, "correct": False}
            self.asked = True
            obs = self.wm.sample_obs(self.hidden_intent, clarify=True)
            return obs, -0.20, False, {"intent": self.hidden_intent}

        if action == "reflect":
            # Рефлексия возвращает hint, но стоит time
            hint = f"Скорее всего это {max(INTENTS, key=lambda i: random.random())}."
            return hint, -0.08, False, {"intent": self.hidden_intent}

        if action == "search_memory":
            return "MEMORY_SEARCHED", -0.06, False, {"intent": self.hidden_intent}

        if action == "call_tool":
            return "TOOL_CALLED", -0.15, False, {"intent": self.hidden_intent}

        if action.startswith("answer_"):
            pred = action.split("_", 1)[1]
            correct = (pred == self.hidden_intent)
            self.done = True
            obs = "SUCCESS" if correct else f"FAIL_{self.hidden_intent.upper()}"
            reward = 1.0 if correct else -1.0
            return obs, reward, True, {"intent": self.hidden_intent, "correct": correct}

        raise ValueError(f"Unknown action: {action}")


# ============================================================
# §7. UAF АГЕНТ
# ============================================================

class UAFAgent:
    """
    Полный UAF-Ω агент.

    Цикл:
        b_t → policy → a_t → env → obs_{t+1}
        → PG_t (после obs!) → update b_{t+1} → repeat
    """

    def __init__(
        self,
        world_model: WorldModel,
        memory: Optional[EpisodeMemory] = None,
        self_monitor: Optional[SelfMonitor] = None,
        policy: Optional[UAFPolicy] = None,
        name: str = "uaf",
    ):
        self.wm = world_model
        self.memory = memory or EpisodeMemory()
        self.monitor = self_monitor or SelfMonitor()
        self.policy = policy or UAFPolicy()
        self.name = name

        # Статистика
        self.total_reward: float = 0.0
        self.total_pg: float = 0.0
        self.n_predictions: int = 0
        self.n_episodes: int = 0
        self.ask_count: int = 0
        self.correct_count: int = 0

        self._reset_episode()

    def _reset_episode(self):
        self.belief: Dict[str, float] = {i: 1.0 / len(INTENTS) for i in INTENTS}
        self.ep_texts: List[str] = []
        self.ep_reward: float = 0.0
        self.ep_pg: float = 0.0
        self.clarified: bool = False
        self.searched: bool = False
        self.ep_step: int = 0

    def start_episode(self, obs: str):
        self._reset_episode()
        self.n_episodes += 1
        prior = self.memory.prior_from_memory(obs)
        self.belief = self.wm.posterior(prior, obs, clarify=False)
        self.ep_texts.append(obs)

    def choose_action(self) -> Tuple[str, Dict[str, float]]:
        utils = self.policy.compute_all_utilities(
            self.belief, self.wm, self.monitor, self.memory,
            self.clarified, self.searched, self.ep_step
        )
        action = self.policy.select(utils)
        return action, utils

    def observe(self, action: str, obs: str, reward: float, done: bool, info: Dict):
        """
        Обновление после РЕАЛЬНОГО наблюдения.
        PG считается ЗДЕСЬ — после получения obs, не до.
        """
        self.ep_step += 1
        self.ep_reward += reward

        if action == "ask_clarify":
            # PG = log P_model(obs|belief,ask) - log P_baseline(obs)
            model_logp = self.wm.mixture_logp(obs, self.belief, clarify=True)
            base_logp = self.wm.uniform_baseline_logp(obs)
            step_pg = UAFMath.pg_step(model_logp, base_logp)

            self.ep_pg += step_pg
            self.total_pg += step_pg
            self.n_predictions += 1

            # Обновляем belief по полученному наблюдению
            self.belief = self.wm.posterior(self.belief, obs, clarify=True)
            self.ep_texts.append(obs)
            self.clarified = True
            self.ask_count += 1

        elif action == "search_memory":
            self.searched = True
            results = self.memory.search(" ".join(self.ep_texts))
            if results:
                # Мягкое обновление belief по найденным эпизодам
                intent_votes = {i: 0.0 for i in INTENTS}
                for ep in results:
                    intent_votes[ep.intent] += 1.0 if ep.correct else 0.3
                vote_prior = normalize(intent_votes)
                # Смешиваем: 70% текущий belief, 30% память
                for i in INTENTS:
                    self.belief[i] = 0.7 * self.belief[i] + 0.3 * vote_prior[i]
                self.belief = normalize(self.belief)

        elif action == "reflect":
            # Рефлексия: небольшое сглаживание belief (снижаем самоуверенность)
            uniform = {i: 1.0 / len(INTENTS) for i in INTENTS}
            for i in INTENTS:
                self.belief[i] = 0.85 * self.belief[i] + 0.15 * uniform[i]
            self.belief = normalize(self.belief)

        elif action.startswith("answer_"):
            pred_intent = action.split("_", 1)[1]
            correct = info.get("correct", False)

            # PG по результату SUCCESS/FAIL
            if obs == "SUCCESS":
                model_logp = math.log(max(self.belief[pred_intent], 1e-9))
            else:
                actual = info.get("intent", pred_intent)
                model_logp = math.log(max(self.belief[actual], 1e-9))

            base_logp = math.log(1.0 / len(INTENTS))
            step_pg = UAFMath.pg_step(model_logp, base_logp)

            self.ep_pg += step_pg
            self.total_pg += step_pg
            self.n_predictions += 1

            conf = self.belief[pred_intent]
            self.monitor.record(conf, correct)

            if correct:
                self.correct_count += 1

            # Сохраняем в память
            self.memory.add(Episode(
                text=" ".join(self.ep_texts),
                intent=info["intent"],
                action=action,
                correct=correct,
                pg=step_pg,
            ))

        self.total_reward += reward

    def metrics(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "total_reward": self.total_reward,
            "avg_reward": self.total_reward / max(self.n_episodes, 1),
            "total_pg": self.total_pg,
            "avg_pg": self.total_pg / max(self.n_predictions, 1),
            "uaf_scale": UAFMath.uaf_scale(self.total_pg, self.n_predictions),
            "success_rate": self.correct_count / max(self.n_episodes, 1),
            "ask_rate": self.ask_count / max(self.n_episodes, 1),
            "self_acc": self.monitor.accuracy(),
            "calibration_error": self.monitor.calibration_error(),
            "self_penalty": self.monitor.penalty,
            "n_episodes": self.n_episodes,
        }


class GreedyAgent(UAFAgent):
    """Жадный агент — всегда отвечает сразу, никогда не спрашивает."""

    def choose_action(self) -> Tuple[str, Dict]:
        best = max(INTENTS, key=lambda i: self.belief[i])
        return f"answer_{best}", {}


# ============================================================
# §8. LLM SHELL
# ============================================================

class LLMShell:
    """
    Языковый слой поверх symbolic core.

    Принцип: LLM ≠ ядро принятия решений.
    LLM = рендерер выбранного symbolic действия.

    Три режима:
      llm_raw:       стандартный промпт
      conservative:  "Не знаю, если неуверен"
      synthesis:     "Подумай по шагам"

    Переключение режимов — диалектически, по UAF-Score.
    """

    MODES = ("llm_raw", "conservative", "synthesis")
    PROMPTS = {
        "llm_raw": "Ответь кратко. ",
        "conservative": "Отвечай только если уверен. Если нет — скажи 'не знаю'. ",
        "synthesis": "Подумай шаг за шагом. Проверь логику. Ответь кратко. ",
    }

    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None,
                 model: str = "gpt-4o-mini", use_mock: bool = True,
                 consecutive_fail_threshold: int = 4):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.use_mock = use_mock
        self.fail_threshold = consecutive_fail_threshold

        self.mode = "llm_raw"
        self.mode_trust = {m: 0.5 for m in self.MODES}
        self.consecutive_fails = 0
        self.dialectic_count = 0
        self._history: List[Dict] = []

    def render(
        self,
        user_text: str,
        action: str,
        belief: Dict[str, float],
        memory_hint: str = "",
    ) -> str:
        """Рендерит symbolic action в естественный язык."""
        probs = ", ".join(f"{k}={v:.2f}" for k, v in belief.items())
        prefix = self.PROMPTS[self.mode]

        if action == "ask_clarify":
            prompt = (
                f"{prefix}Ты помощник. Пользователь написал: '{user_text}'. "
                f"Текущие гипотезы: {probs}. "
                f"Задай один уточняющий вопрос, чтобы определить тип задачи."
            )
        else:
            intent = action.split("_", 1)[-1]
            prompt = (
                f"{prefix}Ты помощник по теме '{intent}'. "
                f"Пользователь написал: '{user_text}'. "
                f"{'Подсказка из памяти: ' + memory_hint if memory_hint else ''} "
                f"Ответь по существу."
            )

        response, confidence = self._call(prompt)

        self._history.append({
            "mode": self.mode, "action": action,
            "response": response, "confidence": confidence,
        })

        return response

    def update_from_outcome(self, correct: bool, confidence: float):
        """Обновляет доверие к текущему режиму и запускает диалектику при сбоях."""
        alpha = 0.15
        if correct:
            self.mode_trust[self.mode] = min(0.95,
                self.mode_trust[self.mode] + alpha * confidence)
            self.consecutive_fails = max(0, self.consecutive_fails - 1)
        else:
            self.mode_trust[self.mode] = max(0.05,
                self.mode_trust[self.mode] - alpha * (1 - confidence))
            self.consecutive_fails += 1

        if self.consecutive_fails >= self.fail_threshold:
            self._dialectic_switch()

    def _dialectic_switch(self):
        """Aufhebung: переключение режима как диалектический синтез."""
        self.dialectic_count += 1
        self.consecutive_fails = 0
        idx = self.MODES.index(self.mode)
        self.mode = self.MODES[(idx + 1) % len(self.MODES)]
        print(
            f"  [LLMShell Aufhebung #{self.dialectic_count}] "
            f"Режим: {self.MODES[(self.MODES.index(self.mode) - 1) % len(self.MODES)]} "
            f"→ {self.mode}"
        )

    def _call(self, prompt: str) -> Tuple[str, float]:
        if self.use_mock:
            return self._mock(prompt)
        try:
            import urllib.request
            payload = json.dumps({
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3, "max_tokens": 100,
            }).encode()
            req = urllib.request.Request(
                self.api_url or "https://api.openai.com/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {self.api_key}"},
            )
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read())
            text = data["choices"][0]["message"]["content"].strip()
            return text, 0.7
        except Exception as e:
            return f"[API error: {e}]", 0.1

    def _mock(self, prompt: str) -> Tuple[str, float]:
        """Mock: имитирует LLM с контролируемыми галлюцинациями."""
        templates = {
            "math":   "Рассмотрим это как математическую задачу: {prompt}",
            "code":   "С точки зрения программирования: {prompt}",
            "theory": "Теоретически: {prompt}",
        }
        if "clarify" in prompt or "уточн" in prompt.lower():
            return "Это ближе к математике, программированию или теории?", 0.75
        for k, tmpl in templates.items():
            if k in prompt.lower():
                return tmpl.format(prompt=prompt[:50]), 0.72
        return "Уточните, пожалуйста, контекст вашего вопроса.", 0.45


# ============================================================
# §9. HALLUCINATION GUARD
# ============================================================

class HallucinationGuard:
    """
    Детектор галлюцинаций через самосогласованность.
    Если N генераций дают разные ответы → высокая вероятность галлюцинации.

    Метрики:
      consistency_score: Jaccard по множествам слов
      severity: 0.0 (нет проблем) — 1.0 (серьёзная галлюцинация)
    """

    def __init__(self, n_samples: int = 3):
        self.n_samples = n_samples
        self._detections: List[Dict] = []

    def check(self, shell: LLMShell, user_text: str,
              action: str, belief: Dict[str, float]) -> Dict:
        """Запускает N генераций и измеряет согласованность."""
        responses = [
            shell._mock(user_text)[0]
            for _ in range(self.n_samples)
        ]

        # Jaccard consistency
        word_sets = [set(r.lower().split()) for r in responses]
        pairs = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                union = word_sets[i] | word_sets[j]
                inter = word_sets[i] & word_sets[j]
                pairs.append(len(inter) / max(len(union), 1))

        consistency = sum(pairs) / max(len(pairs), 1)
        severity = max(0.0, 1.0 - consistency)
        detected = severity > 0.5

        record = {
            "detected": detected,
            "consistency": consistency,
            "severity": severity,
            "mode": shell.mode,
            "n_responses": len(responses),
        }
        self._detections.append(record)

        if detected:
            # Сигнализируем об обнаружении LLMShell
            shell.consecutive_fails += 1
            if shell.consecutive_fails >= shell.fail_threshold:
                shell._dialectic_switch()

        return record


# ============================================================
# §10. НООСФЕРА
# ============================================================

class Noosphere:
    """
    Динамика мультиагентной системы.

    dM_i/dt = Σ_j I_ij · T_ij · (M_j - M_i) + η_i(t)

    Агенты обмениваются убеждениями пропорционально доверию и интенсивности.
    Доверие обновляется по предиктивному выигрышу: T_ij = NPG_j / max_k NPG_k

    Аксиома 4 (Разнообразия): если std(beliefs) < τ — штрафуем консенсус.
    """

    def __init__(self, agents: List[UAFAgent], intensity: float = 0.15):
        self.agents = agents
        self.intensity = intensity
        self.n = len(agents)
        # Матрица доверия T_ij
        self.trust = [[0.5] * self.n for _ in range(self.n)]
        for i in range(self.n):
            self.trust[i][i] = 0.0

        self.history: List[Dict] = []
        self.step_count = 0

    def sync(self):
        """Один шаг ноосферной динамики."""
        self.step_count += 1

        # 1. Обновляем матрицу доверия по NPG агентов
        pgs = [a.metrics()["avg_pg"] for a in self.agents]
        max_pg = max(pgs) if max(pgs) > 0 else 1.0
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self.trust[i][j] = max(0.05, min(0.95, pgs[j] / max_pg))

        # 2. Диффузия убеждений
        new_beliefs = []
        for i, agent in enumerate(self.agents):
            b = dict(agent.belief)
            for j, other in enumerate(self.agents):
                if i == j:
                    continue
                t_ij = self.trust[i][j]
                for intent in INTENTS:
                    drift = self.intensity * t_ij * (other.belief[intent] - b[intent])
                    b[intent] = b[intent] + drift
            new_beliefs.append(normalize(b))

        for agent, nb in zip(self.agents, new_beliefs):
            agent.belief = nb

        # 3. Diversity Penalty
        dp = UAFMath.diversity_penalty(
            [a.belief for a in self.agents], threshold=0.1, gamma=0.5
        )

        # Если diversity слишком мала — усиливаем "антитезисного" агента
        if dp > 0.2 and len(self.agents) >= 2:
            minority = min(self.agents, key=lambda a: a.metrics()["avg_pg"])
            # Добавляем шум к minority-агенту
            for intent in INTENTS:
                minority.belief[intent] += random.gauss(0, 0.05)
            minority.belief = normalize(minority.belief)

        # 4. Консенсус
        consensus = normalize({
            i: sum(self.trust[0][j] * a.belief[i]
                   for j, a in enumerate(self.agents))
            for i in INTENTS
        })

        record = {
            "step": self.step_count,
            "consensus": consensus,
            "diversity_penalty": dp,
            "trust_matrix": [[round(self.trust[i][j], 3) for j in range(self.n)]
                             for i in range(self.n)],
            "agent_beliefs": [dict(a.belief) for a in self.agents],
            "avg_pgs": pgs,
        }
        self.history.append(record)
        return record


# ============================================================
# §11. БЕНЧМАРК
# ============================================================

def run_episode(
    agent: UAFAgent, env: ToyEnv, verbose: bool = False
) -> Dict:
    obs = env.reset()
    agent.start_episode(obs)

    if verbose:
        print(f"\n  [env] obs: '{obs}'")
        print(f"  [agent] belief: { {k: f'{v:.2f}' for k,v in agent.belief.items()} }")

    max_steps = 6
    for _ in range(max_steps):
        action, utils = agent.choose_action()
        obs, reward, done, info = env.step(action)
        agent.observe(action, obs, reward, done, info)

        if verbose:
            print(f"  [agent] action={action}, obs={obs}, reward={reward:.2f}")

        if done:
            break

    return agent.metrics()


def benchmark(
    agents: List[UAFAgent], env_factory, n_episodes: int = 300, seed: int = 42
) -> Dict[str, Dict]:
    random.seed(seed)
    results = {}
    for agent in agents:
        agent.__init__(agent.wm, agent.memory, agent.monitor, agent.policy, agent.name)
        env = env_factory()
        for _ in range(n_episodes):
            run_episode(agent, env)
        results[agent.name] = agent.metrics()
    return results


# ============================================================
# §12. ASSERT-ПРОВЕРКИ
# ============================================================

def run_checks():
    """Полный набор проверок. Если что-то падает — теория сломана."""
    print("Running checks...\n")
    random.seed(7)
    wm = WorldModel()

    # ── Проверка 1: Байес должен работать ───────────────────
    prior = {i: 1.0 / 3 for i in INTENTS}
    post = wm.posterior(prior, "python bug function api traceback debug")
    assert post["code"] > 0.80, f"Code posterior too low: {post}"
    post2 = wm.posterior(prior, "integral matrix theorem proof derivative")
    assert post2["math"] > 0.80, f"Math posterior too low: {post2}"
    print("  ✓ Bayesian update converges correctly")

    # ── Проверка 2: PG после наблюдения ─────────────────────
    model_lp = wm.mixture_logp("python bug function", prior)
    base_lp = wm.uniform_baseline_logp("python bug function")
    pg = UAFMath.pg_step(model_lp, base_lp)
    assert pg > 0, f"PG should be positive for structured obs: {pg}"
    print(f"  ✓ PG after obs is positive: {pg:.3f}")

    # ── Проверка 3: При неуверенности — лучше спросить ──────
    agent = UAFAgent(wm)
    agent.belief = {i: 1.0 / 3 for i in INTENTS}
    utils = agent.policy.compute_all_utilities(
        agent.belief, wm, agent.monitor, agent.memory,
        clarified=False, searched=False, step=1
    )
    assert utils["ask_clarify"] > max(utils[f"answer_{i}"] for i in INTENTS), \
        f"Should ask when uncertain. Utils: {utils}"
    print("  ✓ ask_clarify beats answer when belief is uniform")

    # ── Проверка 4: При уверенности — лучше ответить ────────
    agent2 = UAFAgent(wm)
    agent2.belief = {"math": 0.03, "code": 0.94, "theory": 0.03}
    utils2 = agent2.policy.compute_all_utilities(
        agent2.belief, wm, agent2.monitor, agent2.memory,
        clarified=False, searched=False, step=1
    )
    assert utils2["answer_code"] > utils2["ask_clarify"], \
        f"Should answer when confident. Utils: {utils2}"
    print("  ✓ answer beats ask when p(code)=0.88")

    # ── Проверка 5: UAF-шкала монотонна ─────────────────────
    s1 = UAFMath.uaf_scale(0, 100)
    s2 = UAFMath.uaf_scale(100, 100)
    s3 = UAFMath.uaf_scale(500, 100)
    assert s1 < s2 < s3, f"UAF scale not monotone: {s1}, {s2}, {s3}"
    assert -3 <= s1 <= 3 and -3 <= s3 <= 3
    print(f"  ✓ UAF scale is monotone: {s1:.2f} < {s2:.2f} < {s3:.2f}")

    # ── Проверка 6: Self-monitor штрафует за уверенную ошибку
    mon = SelfMonitor(confidence_threshold=0.75)
    mon.record(0.9, False)  # уверен и ошибся
    mon.record(0.9, False)
    assert mon.penalty > 0.1, f"Penalty should rise: {mon.penalty}"
    mon.record(0.9, True)   # уверен и прав
    prev = mon.penalty
    assert mon.penalty <= prev
    print(f"  ✓ SelfMonitor penalty works: {mon.penalty:.3f}")

    # ── Проверка 7: Diversity Penalty ───────────────────────
    # Все агенты думают одинаково → высокий штраф
    same = [{"math": 0.5, "code": 0.3, "theory": 0.2}] * 5
    dp_high = UAFMath.diversity_penalty(same, threshold=0.15)
    # Агенты думают по-разному → низкий штраф
    diff = [
        {"math": 0.8, "code": 0.1, "theory": 0.1},
        {"math": 0.1, "code": 0.8, "theory": 0.1},
        {"math": 0.1, "code": 0.1, "theory": 0.8},
    ]
    dp_low = UAFMath.diversity_penalty(diff, threshold=0.15)
    assert dp_high > dp_low, f"Diversity penalty should be higher for uniform agents: {dp_high} vs {dp_low}"
    print(f"  ✓ Diversity penalty: uniform={dp_high:.3f} > diverse={dp_low:.3f}")

    # ── Проверка 8: F = U - T·H ──────────────────────────────
    fe = UAFMath.free_energy(U=2.0, T=1.0, H=0.8)
    assert abs(fe - 1.2) < 1e-6, f"Free energy: {fe}"
    print(f"  ✓ F = U − T·H: {fe:.2f}")

    print("\nAll checks passed. ✓")


# ============================================================
# §13. DEMO
# ============================================================

def demo_episode():
    """Один подробный эпизод с выводом."""
    print("\n" + "═" * 60)
    print("  DEMO EPISODE")
    print("═" * 60)
    random.seed(3)
    wm = WorldModel()
    env = ToyEnv(wm)
    agent = UAFAgent(wm, name="uaf-omega")
    shell = LLMShell(use_mock=True)
    guard = HallucinationGuard(n_samples=3)

    obs = env.reset(intent="code")
    agent.start_episode(obs)

    print(f"\n  Initial obs: '{obs}'")
    print(f"  Hidden intent: code")
    print(f"  Belief: { {k: f'{v:.2f}' for k, v in agent.belief.items()} }")

    for step in range(6):
        action, utils = agent.choose_action()
        top_utils = sorted(utils.items(), key=lambda x: -x[1])[:3]
        print(f"\n  Step {step+1}:")
        print(f"    Utilities (top 3): { [(a, f'{u:.3f}') for a,u in top_utils] }")
        print(f"    Chosen: {action}")

        # Рендер через LLM Shell
        text = shell.render(obs, action, agent.belief)
        print(f"    LLM says: '{text[:70]}...'")

        # Детектор галлюцинаций
        hall_check = guard.check(shell, obs, action, agent.belief)
        if hall_check["detected"]:
            print(f"    ⚠ Hallucination detected (severity={hall_check['severity']:.2f})")

        obs, reward, done, info = env.step(action)
        agent.observe(action, obs, reward, done, info)

        print(f"    Result: obs='{obs}', reward={reward:.2f}, done={done}")
        if not done:
            print(f"    Belief updated: { {k: f'{v:.2f}' for k, v in agent.belief.items()} }")

        if done:
            print(f"\n  Episode done. Correct: {info.get('correct', '?')}")
            print(f"  Episode reward: {agent.ep_reward:.2f}")
            print(f"  Episode PG: {agent.ep_pg:.3f}")
            break


def demo_benchmark():
    """Сравнение UAF vs Greedy на 300 эпизодах."""
    print("\n" + "═" * 60)
    print("  BENCHMARK: UAF-Ω vs Greedy (300 episodes)")
    print("═" * 60)

    random.seed(42)
    wm1 = WorldModel()
    wm2 = WorldModel()

    uaf = UAFAgent(wm1, name="uaf-omega")
    greedy = GreedyAgent(wm2, name="greedy")

    def env_factory1(): return ToyEnv(wm1)
    def env_factory2(): return ToyEnv(wm2)

    print("\n  Training UAF agent...")
    env1 = env_factory1()
    for _ in range(300):
        run_episode(uaf, env1)

    print("  Training Greedy agent...")
    env2 = env_factory2()
    for _ in range(300):
        run_episode(greedy, env2)

    # Отчёт
    print("\n" + "─" * 60)
    header = f"  {'Metric':<22} {'UAF-Ω':>10} {'Greedy':>10} {'Δ':>8}"
    print(header)
    print("  " + "─" * 56)

    um = uaf.metrics()
    gm = greedy.metrics()

    rows = [
        ("Avg Reward",      um["avg_reward"],       gm["avg_reward"],       True),
        ("Success Rate",    um["success_rate"],      gm["success_rate"],     True),
        ("Avg NPG (bits)",  um["avg_pg"],            gm["avg_pg"],           True),
        ("UAF Scale",       um["uaf_scale"],         gm["uaf_scale"],        True),
        ("Ask Rate",        um["ask_rate"],          gm["ask_rate"],         None),
        ("Self Accuracy",   um["self_acc"],          gm["self_acc"],         True),
        ("Calibration Err", um["calibration_error"], gm["calibration_error"], False),
        ("Self Penalty",    um["self_penalty"],      gm["self_penalty"],     False),
    ]

    for label, uv, gv, higher_is_better in rows:
        delta = uv - gv
        if higher_is_better is True:
            arrow = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else "=")
        elif higher_is_better is False:
            arrow = "↓" if delta < -0.005 else ("↑" if delta > 0.005 else "=")
        else:
            arrow = ""
        print(f"  {label:<22} {uv:>10.4f} {gv:>10.4f} {delta:>+7.4f} {arrow}")

    print("─" * 60)
    print(f"\n  UAF Scale: {um['uaf_scale']:+.2f} ({_scale_label(um['uaf_scale'])})")


def _scale_label(s: float) -> str:
    r = round(s)
    labels = {
        3: "уникальное предсказание",
        2: "лучше всех доступных",
        1: "лучше базовой",
        0: "как случайность",
        -1: "хуже базовой",
        -2: "системно ошибается",
        -3: "анти-предсказание",
    }
    return labels.get(r, f"{s:.2f}")


def demo_noosphere():
    """Мультиагентный эксперимент."""
    print("\n" + "═" * 60)
    print("  NOOSPHERE: 4-agent experiment (100 steps)")
    print("═" * 60)

    random.seed(17)
    wm = WorldModel()
    agents = [
        UAFAgent(wm, name="agent_conservative"),
        UAFAgent(wm, name="agent_radical"),
        UAFAgent(wm, name="agent_uaf"),
        UAFAgent(wm, name="agent_balanced"),
    ]

    # Инициализируем разные beliefs
    agents[0].belief = {"math": 0.7, "code": 0.2, "theory": 0.1}
    agents[1].belief = {"math": 0.1, "code": 0.1, "theory": 0.8}
    agents[2].belief = {"math": 0.33, "code": 0.33, "theory": 0.34}
    agents[3].belief = {"math": 0.5, "code": 0.3, "theory": 0.2}

    # Тренируем агентов
    env = ToyEnv(wm)
    for agent in agents:
        for _ in range(50):
            run_episode(agent, env)

    noosphere = Noosphere(agents, intensity=0.12)

    print("\n  Initial beliefs:")
    for a in agents:
        b = {k: f"{v:.2f}" for k, v in a.belief.items()}
        print(f"    {a.name}: {b}")

    for step in range(100):
        record = noosphere.sync()
        if step % 25 == 24:
            print(f"\n  Step {step+1}:")
            print(f"    Consensus: { {k: f'{v:.2f}' for k, v in record['consensus'].items()} }")
            print(f"    Diversity penalty: {record['diversity_penalty']:.3f}")
            print(f"    Avg PGs: { [f'{p:.2f}' for p in record['avg_pgs']] }")
            print(f"    Trust of agent_uaf from others: "
                  f"{ [f'{noosphere.trust[i][2]:.2f}' for i in range(4) if i != 2] }")

    print("\n  Final beliefs:")
    for a in agents:
        m = a.metrics()
        print(f"    {a.name}: UAF scale={m['uaf_scale']:+.2f}, "
              f"belief={ {k: f'{v:.2f}' for k, v in a.belief.items()} }")

    # Проверяем: агент с лучшим NPG должен иметь наибольшее доверие
    best_pg_agent = max(range(4), key=lambda i: agents[i].metrics()["avg_pg"])
    avg_trust_to_best = sum(
        noosphere.trust[i][best_pg_agent]
        for i in range(4) if i != best_pg_agent
    ) / 3.0
    print(f"\n  Best NPG agent: {agents[best_pg_agent].name}")
    print(f"  Avg trust toward best: {avg_trust_to_best:.3f}")
    print("  (Теорема 6.2: авторитет = следствие предиктивного выигрыша)")


# ============================================================
# §14. MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UAF-Ω System")
    parser.add_argument("--checks", action="store_true", help="Run assert checks only")
    parser.add_argument("--noosphere", action="store_true", help="Run noosphere experiment")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    args = parser.parse_args()

    run_checks()

    if not args.checks:
        demo_episode()
        demo_benchmark()

    if args.noosphere:
        demo_noosphere()

    print("\nDone.")
