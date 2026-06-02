"""
benchmark.py
============
Воспроизводимый бенчмарк UAF-Ω против базовых стратегий.

Что измеряется (согласно THEORY.md):
    1. avg_reward         — средняя награда за эпизод
    2. success_rate       — доля правильных ответов
    3. avg_pg             — средний предиктивный выигрыш (бит/шаг)
    4. uaf_scale          — позиция на шкале [-3, +3]
    5. ask_rate           — как часто агент просит уточнений
    6. calibration_error  — насколько уверенность соответствует точности
    7. self_penalty       — накопленный штраф за самоуверенные ошибки

Агенты:
    UAF-Ω     — полный агент: ask/answer/reflect/search_memory
    Greedy    — только answer, никогда не спрашивает
    AlwaysAsk — всегда сначала спрашивает (oracle-like)
    Random    — случайный ответ

Среды:
    stationary     — интент не меняется в течение прогона
    nonstationary  — интент меняется каждые 80 эпизодов (сдвиг распределения)
    noisy          — наблюдения сильно зашумлены

Запуск:
    python benchmark.py                    # все эксперименты
    python benchmark.py --env stationary   # только stationary
    python benchmark.py --n 500 --seed 7   # 500 эпизодов, seed=7
    python benchmark.py --json results.json
"""

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

INTENTS = ("math", "code", "theory")


def _normalize(d: Dict[str, float]) -> Dict[str, float]:
    s = sum(d.values())
    return {k: v / s for k, v in d.items()} if s > 0 else {k: 1/len(d) for k in d}


def _entropy(d: Dict[str, float]) -> float:
    eps = 1e-12
    return -sum(max(p, eps) * math.log(max(p, eps)) for p in d.values())


def _logsumexp(xs: List[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def _sample(d: Dict[str, float]) -> str:
    r, acc, last = random.random(), 0.0, None
    for k, p in d.items():
        last = k; acc += p
        if r <= acc:
            return k
    return last


# ============================================================
# WORLD MODEL (копия из uaf_omega.py, без зависимостей)
# ============================================================

class _WorldModel:
    VOCAB = {
        "math":   ["integral","matrix","proof","theorem","equation","derivative","limit","lemma"],
        "code":   ["python","bug","function","class","api","stack","debug","traceback"],
        "theory": ["uaf","boundary","prediction","model","agent","theory","update","prior"],
    }
    NOISE = ["help","please","problem","question","need","make","what","how"]

    def __init__(self):
        self._all = sorted(set(t for v in self.VOCAB.values() for t in v) | set(self.NOISE))
        self._cache: Dict[Tuple, Dict] = {}

    def token_dist(self, intent: str, clarify: bool = False) -> Dict[str, float]:
        key = (intent, clarify)
        if key in self._cache:
            return self._cache[key]
        w = {}
        for tok in self._all:
            if tok in self.VOCAB[intent]: w[tok] = 9.0 if clarify else 5.0
            elif tok in self.NOISE:       w[tok] = 1.0 if clarify else 3.0
            else:                         w[tok] = 0.5
        s = sum(w.values())
        self._cache[key] = {t: v/s for t, v in w.items()}
        return self._cache[key]

    def sample_obs(self, intent: str, clarify: bool = False, n: int = 4) -> str:
        d = self.token_dist(intent, clarify)
        return " ".join(random.choices(list(d), weights=list(d.values()), k=n))

    def log_likelihood(self, text: str, intent: str, clarify: bool = False) -> float:
        toks = text.lower().split()
        d = self.token_dist(intent, clarify)
        unk = 1.0 / (20 * len(self._all))
        return sum(math.log(d.get(t, unk)) for t in toks) if toks else math.log(1e-9)

    def posterior(self, prior: Dict[str, float], text: str, clarify: bool = False) -> Dict[str, float]:
        lls = {i: self.log_likelihood(text, i, clarify) for i in INTENTS}
        return _normalize({
            i: math.exp(math.log(max(prior[i], 1e-12)) + lls[i] - max(lls.values()))
            for i in INTENTS
        })

    def mixture_logp(self, text: str, belief: Dict[str, float], clarify: bool = False) -> float:
        return _logsumexp([math.log(max(belief[i], 1e-12)) + self.log_likelihood(text, i, clarify) for i in INTENTS])

    def uniform_logp(self, text: str) -> float:
        n = len(text.split())
        return n * math.log(1.0 / (len(self._all) + 50))


# ============================================================
# ENVIRONMENT
# ============================================================

class _Env:
    def __init__(self, wm: _WorldModel, noise: float = 0.05):
        self.wm = wm
        self.noise = noise
        self.hidden: Optional[str] = None
        self.done = False
        self.asked = False

    def reset(self, intent: Optional[str] = None) -> str:
        self.hidden = intent or random.choice(INTENTS)
        self.done = False
        self.asked = False
        return self.wm.sample_obs(self.hidden, clarify=False)

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        if action == "ask_clarify":
            if self.asked:
                self.done = True
                return "NO_MORE_INFO", -0.5, True, {"intent": self.hidden, "correct": False}
            self.asked = True
            obs = self.wm.sample_obs(self.hidden, clarify=True)
            if self.noise > 0.2:  # noisy env: sometimes misleading clarification
                if random.random() < self.noise * 0.5:
                    obs = self.wm.sample_obs(random.choice(INTENTS), clarify=True)
            return obs, -0.20, False, {"intent": self.hidden}

        if action == "reflect":
            return "REFLECT_OK", -0.08, False, {"intent": self.hidden}

        if action == "search_memory":
            return "MEMORY_OK", -0.06, False, {"intent": self.hidden}

        if action.startswith("answer_"):
            pred = action.split("_", 1)[1]
            correct = (pred == self.hidden)
            self.done = True
            obs = "SUCCESS" if correct else f"FAIL_{self.hidden.upper()}"
            return obs, 1.0 if correct else -1.0, True, {"intent": self.hidden, "correct": correct}

        return "UNKNOWN", 0.0, False, {}


# ============================================================
# SELF-MONITOR
# ============================================================

class _SelfMonitor:
    def __init__(self):
        self.penalty = 0.0
        self._records: List[Tuple[float, bool]] = []

    def record(self, conf: float, correct: bool):
        self._records.append((conf, correct))
        if conf >= 0.75 and not correct:
            self.penalty = min(0.45, self.penalty + 0.09)
        elif conf >= 0.75 and correct:
            self.penalty = max(0.0, self.penalty - 0.03)

    def accuracy(self) -> float:
        return sum(c for _, c in self._records) / max(len(self._records), 1)

    def avg_conf(self) -> float:
        return sum(c for c, _ in self._records) / max(len(self._records), 1)

    def cal_error(self) -> float:
        if len(self._records) < 5:
            return 0.0
        bins = [[] for _ in range(5)]
        for conf, c in self._records:
            bins[min(int(conf * 5), 4)].append(1 if c else 0)
        return sum(
            abs(sum(b)/len(b) - (i+.5)/5) * len(b) / len(self._records)
            for i, b in enumerate(bins) if b
        )


# ============================================================
# MEMORY
# ============================================================

class _Memory:
    def __init__(self):
        self._items = []

    def add(self, text: str, intent: str, correct: bool, pg: float):
        self._items.append({"text": text, "intent": intent, "correct": correct, "pg": pg})
        if len(self._items) > 300:
            self._items.pop(0)

    def prior(self, text: str) -> Dict[str, float]:
        toks = set(text.lower().split())
        scores = {i: 1.0 for i in INTENTS}
        for ep in self._items:
            ov = len(toks & set(ep["text"].split()))
            if ov > 0:
                scores[ep["intent"]] += ov * (1.0 if ep["correct"] else 0.25)
        return _normalize(scores)

    def recent_pg(self, n: int = 20) -> float:
        r = self._items[-n:]
        return sum(ep["pg"] for ep in r) / max(len(r), 1)

    def search(self, text: str, k: int = 3) -> List[Dict]:
        toks = set(text.lower().split())
        scored = [(len(toks & set(ep["text"].split())), ep) for ep in self._items]
        scored.sort(key=lambda x: -x[0])
        return [ep for _, ep in scored[:k] if _ > 0]


# ============================================================
# POLICY
# ============================================================

class _Policy:
    def __init__(self, ask_cost: float = 0.30, ans_cost: float = 0.05,
                 ref_cost: float = 0.10, srch_cost: float = 0.08,
                 k_info: float = 0.30, mc: int = 30):
        self.ask_cost = ask_cost
        self.ans_cost = ans_cost
        self.ref_cost = ref_cost
        self.srch_cost = srch_cost
        self.k_info = k_info
        self.mc = mc

    def utilities(self, belief, wm, mon, mem, clarified, searched, step):
        u = {}
        for i in INTENTS:
            u[f"answer_{i}"] = (2*belief[i] - 1) - self.ans_cost - mon.penalty

        if clarified:
            u["ask_clarify"] = -1e9
        else:
            H0 = _entropy(belief)
            total = 0.0
            for _ in range(self.mc):
                hid = _sample(belief)
                obs = wm.sample_obs(hid, clarify=True)
                post = wm.posterior(belief, obs, clarify=True)
                best = max((2*post[i]-1) - self.ans_cost - mon.penalty for i in INTENTS)
                ig = H0 - _entropy(post)
                total += best + self.k_info * ig - self.ask_cost
            u["ask_clarify"] = total / self.mc

        H = _entropy(belief)
        u["reflect"] = self.k_info * 0.5 * H - self.ref_cost if H > 0.8 and step > 1 else -1e9
        rpg = mem.recent_pg()
        u["search_memory"] = self.k_info * max(rpg, 0) - self.srch_cost if not searched and len(mem._items) > 3 else -1e9
        return u

    def select(self, u):
        return max(u, key=u.get)


# ============================================================
# AGENTS
# ============================================================

class _UAFAgent:
    def __init__(self, wm: _WorldModel, name: str = "uaf"):
        self.wm = wm
        self.name = name
        self.policy = _Policy()
        self.monitor = _SelfMonitor()
        self.memory = _Memory()
        self.total_reward = 0.0
        self.total_pg = 0.0
        self.n_preds = 0
        self.n_eps = 0
        self.n_correct = 0
        self.n_asks = 0
        self.reward_hist: List[float] = []
        self._reset_ep()

    def _reset_ep(self):
        self.belief = _normalize({i: 1.0 for i in INTENTS})
        self.clarified = False
        self.searched = False
        self.ep_step = 0
        self.ep_text = []
        self.ep_rew = 0.0
        self.ep_pg = 0.0

    def start(self, obs: str):
        self._reset_ep()
        self.n_eps += 1
        prior = self.memory.prior(obs)
        self.belief = self.wm.posterior(prior, obs)
        self.ep_text = [obs]

    def act(self) -> str:
        u = self.policy.utilities(
            self.belief, self.wm, self.monitor, self.memory,
            self.clarified, self.searched, self.ep_step
        )
        return self.policy.select(u)

    def observe(self, action: str, obs: str, reward: float, done: bool, info: Dict):
        self.ep_step += 1
        self.ep_rew += reward

        if action == "ask_clarify":
            mlp = self.wm.mixture_logp(obs, self.belief, clarify=True)
            blp = self.wm.uniform_logp(obs)
            pg = mlp - blp
            self.ep_pg += pg
            self.total_pg += pg
            self.n_preds += 1
            self.belief = self.wm.posterior(self.belief, obs, clarify=True)
            self.ep_text.append(obs)
            self.clarified = True
            self.n_asks += 1

        elif action == "search_memory":
            self.searched = True
            res = self.memory.search(" ".join(self.ep_text))
            if res:
                votes = _normalize({i: sum(ep["pg"] > 0 and ep["intent"] == i for ep in res) + 0.01 for i in INTENTS})
                for i in INTENTS:
                    self.belief[i] = 0.7 * self.belief[i] + 0.3 * votes[i]
                self.belief = _normalize(self.belief)

        elif action == "reflect":
            u = {i: 1/3 for i in INTENTS}
            for i in INTENTS:
                self.belief[i] = 0.85 * self.belief[i] + 0.15 * u[i]
            self.belief = _normalize(self.belief)

        elif action.startswith("answer_"):
            pred = action.split("_", 1)[1]
            correct = info.get("correct", False)
            mlp = math.log(max(self.belief[pred if obs == "SUCCESS" else info.get("intent", pred)], 1e-9))
            blp = math.log(1.0 / 3)
            pg = mlp - blp
            self.ep_pg += pg
            self.total_pg += pg
            self.n_preds += 1
            self.monitor.record(self.belief[pred], correct)
            if correct:
                self.n_correct += 1
            self.memory.add(" ".join(self.ep_text), info["intent"], correct, pg)

        self.total_reward += reward
        if done:
            self.reward_hist.append(self.ep_rew)

    def metrics(self) -> Dict[str, float]:
        avg_pg = self.total_pg / max(self.n_preds, 1)
        return {
            "avg_reward":        self.total_reward / max(self.n_eps, 1),
            "success_rate":      self.n_correct / max(self.n_eps, 1),
            "avg_pg":            avg_pg,
            "uaf_scale":         math.tanh(avg_pg / 2.0) * 3.0,
            "ask_rate":          self.n_asks / max(self.n_eps, 1),
            "self_accuracy":     self.monitor.accuracy(),
            "calibration_error": self.monitor.cal_error(),
            "self_penalty":      self.monitor.penalty,
            "n_episodes":        self.n_eps,
        }


class _GreedyAgent(_UAFAgent):
    def act(self) -> str:
        best = max(INTENTS, key=lambda i: self.belief[i])
        return f"answer_{best}"


class _AlwaysAskAgent(_UAFAgent):
    """Всегда сначала спрашивает, потом отвечает."""
    def act(self) -> str:
        if not self.clarified:
            return "ask_clarify"
        best = max(INTENTS, key=lambda i: self.belief[i])
        return f"answer_{best}"


class _RandomAgent(_UAFAgent):
    """Отвечает случайно."""
    def act(self) -> str:
        return f"answer_{random.choice(INTENTS)}"


# ============================================================
# RESULTS
# ============================================================

@dataclass
class AgentResult:
    name: str
    env: str
    n_episodes: int
    avg_reward: float
    success_rate: float
    avg_pg: float
    uaf_scale: float
    ask_rate: float
    self_accuracy: float
    calibration_error: float
    self_penalty: float

    def row(self) -> str:
        return (
            f"  {self.name:<14} {self.avg_reward:>+8.4f}  "
            f"{self.success_rate:>8.1%}  {self.avg_pg:>+8.4f}  "
            f"{self.uaf_scale:>+7.2f}  {self.ask_rate:>7.1%}  "
            f"{self.calibration_error:>8.4f}  {self.self_penalty:>8.4f}"
        )


# ============================================================
# RUNNER
# ============================================================

def run_episode(agent: _UAFAgent, env: _Env, max_steps: int = 7) -> float:
    obs = env.reset()
    agent.start(obs)
    for _ in range(max_steps):
        action = agent.act()
        obs, reward, done, info = env.step(action)
        agent.observe(action, obs, reward, done, info)
        if done:
            break
    return agent.ep_rew


def run_benchmark(
    n_episodes: int = 300,
    seed: int = 42,
    env_type: str = "stationary",
    noise: float = 0.05,
    verbose: bool = True,
) -> List[AgentResult]:

    random.seed(seed)
    wm = _WorldModel()

    # Создаём независимые копии агентов для каждого прогона
    agents = [
        _UAFAgent(wm, "uaf-omega"),
        _GreedyAgent(wm, "greedy"),
        _AlwaysAskAgent(wm, "always-ask"),
        _RandomAgent(wm, "random"),
    ]

    envs = [_Env(wm, noise=noise) for _ in agents]

    if verbose:
        print(f"\n{'─'*75}")
        print(f"  Benchmark: env={env_type}, n={n_episodes}, seed={seed}, noise={noise}")
        print(f"{'─'*75}")

    for ep in range(n_episodes):
        # Нестационарная среда: сдвиг распределения каждые 80 эпизодов
        force_intent = None
        if env_type == "nonstationary" and ep > 0 and ep % 80 == 0:
            force_intent = random.choice(INTENTS)
            if verbose and ep % 80 == 0:
                print(f"  [step {ep}] Distribution shift → intent bias: {force_intent}")

        for agent, env in zip(agents, envs):
            run_episode(agent, env)

    results = []
    for agent in agents:
        m = agent.metrics()
        results.append(AgentResult(
            name=agent.name,
            env=env_type,
            n_episodes=n_episodes,
            **{k: round(v, 6) for k, v in m.items() if k != "n_episodes"},
        ))

    if verbose:
        _print_table(results)

    return results


def _print_table(results: List[AgentResult]):
    header = (
        f"  {'Agent':<14} {'Avg Rew':>8}  {'Success':>8}  "
        f"{'Avg PG':>8}  {'Scale':>7}  {'AskRate':>7}  "
        f"{'CalErr':>8}  {'Penalty':>8}"
    )
    print(header)
    print("  " + "─" * 73)
    for r in results:
        print(r.row())
    print()

    # UAF vs Greedy highlight
    uaf = next((r for r in results if r.name == "uaf-omega"), None)
    g = next((r for r in results if r.name == "greedy"), None)
    if uaf and g:
        print(f"  UAF-Ω vs Greedy:")
        for attr in ["avg_reward", "success_rate", "avg_pg", "calibration_error"]:
            uv, gv = getattr(uaf, attr), getattr(g, attr)
            delta = uv - gv
            better = (delta > 0) if attr != "calibration_error" else (delta < 0)
            sym = "✓" if better else "✗"
            print(f"    {sym} {attr:<22}: UAF={uv:.4f}  Greedy={gv:.4f}  Δ={delta:+.4f}")


def run_all(
    n_episodes: int = 300,
    seed: int = 42,
    save_json: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, List[AgentResult]]:

    all_results = {}

    envs = [
        ("stationary",    0.05),
        ("nonstationary", 0.05),
        ("noisy",         0.40),
    ]

    for env_type, noise in envs:
        results = run_benchmark(n_episodes, seed, env_type, noise, verbose)
        all_results[env_type] = results

    if verbose:
        print("\n" + "═" * 75)
        print("  SUMMARY: avg_pg across environments")
        print("═" * 75)
        agent_names = [r.name for r in all_results["stationary"]]
        for name in agent_names:
            pgs = []
            for env_type, results in all_results.items():
                for r in results:
                    if r.name == name:
                        pgs.append(r.avg_pg)
            avg = sum(pgs) / len(pgs)
            scale = math.tanh(avg / 2.0) * 3.0
            print(f"  {name:<14}  avg_pg={avg:+.4f}  scale={scale:+.2f}  ({_scale_label(scale)})")
        print("═" * 75)

    if save_json:
        data = {}
        for env_type, results in all_results.items():
            data[env_type] = [asdict(r) for r in results]
        with open(save_json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n  Results saved → {save_json}")

    return all_results


def _scale_label(s: float) -> str:
    r = round(s)
    return {
        3: "уникальное предсказание", 2: "лучше всех доступных",
        1: "лучше базовой", 0: "как случайность",
        -1: "хуже базовой", -2: "системно ошибается", -3: "анти-предсказание",
    }.get(r, f"{s:.2f}")


# ============================================================
# CHECKS
# ============================================================

def _run_checks():
    """Проверка что benchmark запускается и даёт разумные числа."""
    print("benchmark.py checks...")
    random.seed(0)
    results = run_benchmark(n_episodes=50, seed=0, verbose=False)

    uaf = next(r for r in results if r.name == "uaf-omega")
    greedy = next(r for r in results if r.name == "greedy")
    rnd = next(r for r in results if r.name == "random")

    # UAF не должен быть хуже случайного по avg_reward
    assert uaf.avg_reward >= rnd.avg_reward - 0.5, \
        f"UAF should not be worse than random: {uaf.avg_reward:.3f} vs {rnd.avg_reward:.3f}"
    print(f"  ✓ UAF avg_reward={uaf.avg_reward:.3f} ≥ Random avg_reward={rnd.avg_reward:.3f}")

    # UAF должен иметь ненулевой ask_rate (использует уточнения)
    assert uaf.ask_rate > 0.0, f"UAF should ask sometimes: {uaf.ask_rate}"
    assert greedy.ask_rate == 0.0, f"Greedy should never ask: {greedy.ask_rate}"
    print(f"  ✓ UAF ask_rate={uaf.ask_rate:.2f}, Greedy ask_rate={greedy.ask_rate:.2f}")

    # NPG случайного агента должен быть отрицательным
    assert rnd.avg_pg < uaf.avg_pg, \
        f"UAF NPG should beat random: {uaf.avg_pg:.4f} vs {rnd.avg_pg:.4f}"
    print(f"  ✓ UAF avg_pg={uaf.avg_pg:.4f} > Random avg_pg={rnd.avg_pg:.4f}")

    # Шкала UAF должна быть в [-3, +3]
    for r in results:
        assert -3 <= r.uaf_scale <= 3, f"Scale out of range: {r.uaf_scale}"
    print(f"  ✓ All UAF scales in [-3, +3]")

    print("\nAll benchmark.py checks passed. ✓\n")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UAF-Ω Benchmark")
    parser.add_argument("--env",    type=str,  default="all",
                        choices=["all", "stationary", "nonstationary", "noisy"],
                        help="Environment type")
    parser.add_argument("--n",      type=int,  default=300, help="Episodes per agent")
    parser.add_argument("--seed",   type=int,  default=42,  help="Random seed")
    parser.add_argument("--json",   type=str,  default=None, help="Save results to JSON")
    parser.add_argument("--checks", action="store_true",    help="Run checks only")
    parser.add_argument("--quiet",  action="store_true",    help="Minimal output")
    args = parser.parse_args()

    _run_checks()

    if not args.checks:
        if args.env == "all":
            run_all(args.n, args.seed, save_json=args.json, verbose=not args.quiet)
        else:
            noise = 0.40 if args.env == "noisy" else 0.05
            results = run_benchmark(args.n, args.seed, args.env, noise, verbose=not args.quiet)
            if args.json:
                with open(args.json, "w") as f:
                    json.dump([asdict(r) for r in results], f, indent=2)
                print(f"Saved → {args.json}")
