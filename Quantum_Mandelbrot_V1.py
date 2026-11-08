# Quantum_Logistic_Mandelbrot



"""
Quantum_Logistic_Mandelbrot_V1.py

Python 3.11
Qiskit 1.4.4

OSNOVNA DINAMIKA
---------------
    x_next = r*x*(1-x)

Konjugovana kvadratna mapa:
    z = r*(1/2-x)
    c = r*(2-r)/4
    z_next = z*z+c

Koriste se realni parametri logisticke porodice.
Mandelbrotova veza odredjuje istu porodicu kvadratnih mapa.

POSEBNOST MODELA
---------------
Model uci raspodelu NEUREDJENOG skupa od sedam brojeva.

Ne postoje zasebni modeli za sedam sortiranih pozicija.

Istorija odredjuje amplitude zajednickog ulaznog stanja:
    sest kubita za x,
    cetiri kubita za r.

Posle naucenih RY rotacija primenjuje se CNOT lanac.
Time se povezuju latentno stanje i parametar dinamike.

Kvadratni orakl prenosi celu Bornovu raspodelu:
    |r,x,y> -> |r,x,y XOR f_r(x)>.

Razliciti ulazi koji daju isti izlaz sabiraju svoje mase.
Ulaz i parametar ostaju sacuvani u pomocnim registrima.

PRECIZNOST
----------
x ima 64 sredista celija u intervalu (0,1).
r ima 16 unapred navedenih vrednosti.
Uce se njihove zajednicke amplitude uslovljene istorijom.

Mapa se iterira bez medjukoracnog zaokruzivanja.
Tek konacni rezultat kodira se u 39 izlaznih celija.
Vrednost 1 pripada poslednjoj celiji.

QISKIT
------
Zajednicka konstrukcija jedne kopije ima:
    6 ulaznih,
    4 parametarska,
    6 izlaznih kubita.

Simulacija je blokovska:
    izvorno stanje simulira se sa 10 kubita,
    svaki r-blok sa 12 kubita.

Trag preko sacuvanog r registra tacno odgovara zbiru
tih blokova. Nema slucajnog uzorkovanja.

Sedam izlaznih sestokubitnih registara predstavlja
sedam nezavisnih kopija istog naucenog kanala.

SETOVI VELICINE SEDAM
--------------------
Posle uslovljavanja na razlicite izlaze:

    P(S | history) = product_{v in S} p(v) / e_7(p)

e_7 je elementarni simetricni polinom sedmog stepena.

To je nativna raspodela skupova velicine sedam.
Duplikati se ne popravljaju posle predikcije.

Globalni MAP je skup sedam najvecih modelskih p(v).
To nisu rangirane istorijske ucestalosti.

UCENJE
------
L-BFGS-B optimizuje parametre kvantnih rotacija.
Koristi se tacan uslovni NLL skupova velicine sedam.
Gradijent prolazi kroz Bornove mase, kvadratni orakl
i normalizator e_7.

Obuka koristi tacnu algebraicku simulaciju istog kola.
Qiskit proverava izvorno stanje i izvodi zavrsni orakl.

Hronoloska validacija bira prozor istorije i broj iteracija.
Test je odvojen.
Zavrsni model koristi sve istorijske prelaze.

Nema klasifikatora, slucajnih kombinacija, dodatog suma,
normalne prediktivne raspodele ili kozmetickih filtera.

ISPISI
------
DIAGNOSTICS=False:
    naziv i posebnost,
    rezultat validacije i testa,
    NEXT za svaki CSV,
    materijalna upozorenja.
"""

import csv
import hashlib
import itertools
import math
import random
from pathlib import Path

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate, StatePreparation
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
from scipy.special import logsumexp


random.seed(39)
np.random.seed(39)


FILES = (
    Path("/Users/4c/Desktop/GHQ/data/loto7_4716_k89.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4716_k89_loto_2980.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4716_k89_loto_plus_1736.csv"),
)

DIAGNOSTICS = False

N = 39
K = 7

X_QUBITS = 6
R_QUBITS = 4
OUTPUT_QUBITS = 6

X_DIM = 1 << X_QUBITS
R_DIM = 1 << R_QUBITS
OUTPUT_DIM = 1 << OUTPUT_QUBITS

SOURCE_QUBITS = X_QUBITS + R_QUBITS
SOURCE_DIM = 1 << SOURCE_QUBITS

WINDOWS = (1, 2)
MAP_STEPS = (1, 2)

REGULARIZATION = 0.001
RESTARTS = 2
MAX_ITERATIONS = 160
GRADIENT_TOLERANCE = 1e-5
BATCH_SIZE = 64

TRAIN_FRACTION = 0.70
VALIDATION_END_FRACTION = 0.85

R_VALUES = np.array([
    0.25,
    0.50,
    1.00,
    1.50,
    2.00,
    2.50,
    3.00,
    1.0 + np.sqrt(6.0),
    3.50,
    3.544090359551923,
    3.569945672,
    3.65,
    3.75,
    1.0 + 2.0 * np.sqrt(2.0),
    3.90,
    4.00,
], dtype=np.float64)

X_VALUES = (
    np.arange(X_DIM, dtype=np.float64) + 0.5
) / X_DIM

RAW_BASIS = np.arange(SOURCE_DIM, dtype=np.int64)

RAW_BITS = (
    RAW_BASIS[:, None]
    >> np.arange(SOURCE_QUBITS, dtype=np.int64)[None, :]
) & 1

# Preslikavanje baznih indeksa kroz CNOT 0->1->...->9.
SOURCE_PERMUTATION = RAW_BASIS.copy()

for control in range(SOURCE_QUBITS - 1):
    SOURCE_PERMUTATION ^= (
        ((SOURCE_PERMUTATION >> control) & 1)
        << (control + 1)
    )

SOURCE_INVERSE = np.argsort(SOURCE_PERMUTATION)


class DistributionError(ArithmeticError):
    pass


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def detail(*values):
    if DIAGNOSTICS:
        print(*values, flush=True)


def load_csv(path):
    raw = path.read_bytes()
    rows = []

    for line_number, row in enumerate(
        csv.reader(raw.decode("utf-8-sig").splitlines()),
        start=1,
    ):
        if not row or all(not cell.strip() for cell in row):
            continue

        require(
            len(row) == K,
            f"{path.name}, red {line_number}: potrebno je sedam kolona.",
        )

        try:
            values = [int(cell.strip()) for cell in row]
        except ValueError as exc:
            raise RuntimeError(
                f"{path.name}, red {line_number}: necelobrojna vrednost."
            ) from exc

        require(
            len(set(values)) == K
            and all(1 <= value <= N for value in values)
            and values == sorted(values),
            f"{path.name}, red {line_number}: nelegalna kombinacija.",
        )
        rows.append(values)

    require(len(rows) >= 5, f"{path.name}: nedovoljno kola.")

    return {
        "path": path,
        "data": np.asarray(rows, dtype=np.int64) - 1,
        "digest": hashlib.sha256(raw).hexdigest(),
    }


def build_specification(steps):
    require(
        isinstance(steps, int) and steps >= 1,
        "Neispravan broj iteracija mape.",
    )
    require(
        len(R_VALUES) == R_DIM
        and np.all((R_VALUES > 0.0) & (R_VALUES <= 4.0)),
        "Neispravna mreza logistickih parametara.",
    )

    values = np.broadcast_to(
        X_VALUES[None, :], (R_DIM, X_DIM)
    ).copy()

    for _ in range(steps):
        values = R_VALUES[:, None] * values * (1.0 - values)

    require(
        np.isfinite(values).all()
        and np.all((values >= 0.0) & (values <= 1.0)),
        "Logisticka iteracija je napustila interval [0,1].",
    )

    # Definicija izlaznih celija. Tacna krajnja vrednost 1
    # pripada poslednjoj celiji, a ne indeksu 39.
    table = np.floor(N * values).astype(np.int64)
    table[table == N] = N - 1

    require(
        np.all((table >= 0) & (table < N)),
        "Izlaz mape nije legalan bazni indeks.",
    )

    latent_r = SOURCE_PERMUTATION >> X_QUBITS
    latent_x = SOURCE_PERMUTATION & (X_DIM - 1)

    lookup = table[latent_r, latent_x]
    support = np.bincount(lookup, minlength=N) > 0

    phase_gates = []

    for parameter in range(R_DIM):
        gates = []

        for output_bit in range(OUTPUT_QUBITS):
            diagonal = np.ones(2 * X_DIM, dtype=np.complex128)
            diagonal[X_DIM:] = (
                1.0 - 2.0 * ((table[parameter] >> output_bit) & 1)
            )
            gates.append(DiagonalGate(diagonal.tolist()))

        phase_gates.append(tuple(gates))

    return {
        "steps": steps,
        "table": table,
        "lookup": lookup,
        "support": support,
        "phase_gates": tuple(phase_gates),
    }


def raw_history_features(data, times, window):
    """
    Istorijski ulaz je prisustvo brojeva u prethodnim kolima.
    Cilj modela je takodje neuredjen skup, bez sedam slot-modela.
    """
    times = np.asarray(times, dtype=np.int64)

    require(
        window in WINDOWS
        and times.ndim == 1
        and len(times) > 0
        and np.all((times >= 1) & (times <= len(data))),
        "Neispravan istorijski prozor ili vreme.",
    )

    features = np.zeros(
        (len(times), N * window + window - 1),
        dtype=np.float64,
    )

    for lag in range(1, window + 1):
        source = times - lag
        available_rows = np.flatnonzero(source >= 0)
        start = N * (lag - 1)

        if len(available_rows):
            features[
                available_rows[:, None],
                start + data[source[available_rows]],
            ] = 1.0

        if lag >= 2:
            features[:, N * window + lag - 2] = (
                source >= 0
            ).astype(float)

    return features


def fit_transform(raw, training_rows):
    require(
        1 <= training_rows <= len(raw),
        "Neispravna granica skaliranja.",
    )

    training = raw[:training_rows]
    mean = training.mean(axis=0)
    scale = training.std(axis=0)

    return {
        "mean": mean,
        "scale": scale,
        "keep": scale > 1e-12,
    }


def transform_features(raw, transform):
    keep = transform["keep"]

    features = np.ones(
        (len(raw), 1 + int(np.sum(keep))),
        dtype=np.float64,
    )

    if np.any(keep):
        features[:, 1:] = (
            raw[:, keep] - transform["mean"][keep]
        ) / transform["scale"][keep]

    require(
        np.isfinite(features).all(),
        "Nekonacne istorijske karakteristike.",
    )
    return features


def initial_parameters(feature_count, restart):
    index = np.arange(
        SOURCE_QUBITS * feature_count,
        dtype=np.float64,
    ).reshape(SOURCE_QUBITS, feature_count)

    parameters = (
        0.025
        * np.sin(index + 1.0 + 0.71 * restart)
        / np.sqrt(feature_count)
    )

    qubits = np.arange(1, SOURCE_QUBITS + 1, dtype=float)

    if restart == 0:
        parameters[:, 0] = (
            np.pi / 2.0 + 0.17 * np.sin(qubits)
        )
    else:
        parameters[:, 0] = (
            1.9 + 0.20 * np.cos(qubits * np.sqrt(2.0))
        )

    return parameters.ravel()


def regularization_mask(feature_count):
    mask = np.ones(
        (SOURCE_QUBITS, feature_count),
        dtype=np.float64,
    )
    mask[:, 0] = 0.0
    return mask.ravel()


def forward_probabilities(theta, features, specification, record=False):
    weights = np.asarray(theta, dtype=float).reshape(
        SOURCE_QUBITS, features.shape[1]
    )
    angles = features @ weights.T

    sine_squared = np.sin(angles / 2.0) ** 2
    cosine_squared = np.cos(angles / 2.0) ** 2

    factors = np.where(
        RAW_BITS[None, :, :] == 1,
        sine_squared[:, None, :],
        cosine_squared[:, None, :],
    )

    prefix = None
    suffix = None

    if record:
        prefix = np.ones(
            (len(features), SOURCE_DIM, SOURCE_QUBITS + 1),
            dtype=np.float64,
        )
        suffix = np.ones_like(prefix)

        for bit in range(SOURCE_QUBITS):
            prefix[:, :, bit + 1] = (
                prefix[:, :, bit] * factors[:, :, bit]
            )

        for bit in range(SOURCE_QUBITS - 1, -1, -1):
            suffix[:, :, bit] = (
                suffix[:, :, bit + 1] * factors[:, :, bit]
            )

        source_probabilities = prefix[:, :, -1]
    else:
        source_probabilities = np.prod(factors, axis=2)

    output = np.zeros((len(features), N), dtype=np.float64)

    # Tacan prenos mase kroz mnogo-na-jedan kvadratnu mapu.
    np.add.at(
        output,
        (
            np.arange(len(features))[:, None],
            specification["lookup"][None, :],
        ),
        source_probabilities,
    )

    require(
        np.isfinite(output).all()
        and np.all(output >= 0.0)
        and np.allclose(
            output.sum(axis=1),
            1.0,
            atol=2e-10,
            rtol=0.0,
        ),
        "Izlazna Bornova raspodela nije normalizovana.",
    )

    return output, {
        "angles": angles,
        "prefix": prefix,
        "suffix": suffix,
    }


def safe_log(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    result = np.full(probabilities.shape, -np.inf)
    positive = probabilities > 0.0
    result[positive] = np.log(probabilities[positive])
    return result


def set_statistics(log_probabilities, cardinality=K, targets=None):
    """
    e_k(p) = sum_{|S|=k} product_{v in S} p(v)

    Ukljucenost broja:
        pi(v) = p(v) * e_{k-1}(p bez v) / e_k(p)

    Zbir ukljucenosti je k, a ne jedan.
    """
    logs = np.asarray(log_probabilities, dtype=np.float64)

    require(
        logs.ndim == 2
        and not np.isnan(logs).any()
        and not np.isposinf(logs).any(),
        "Neispravni logaritmi raspodele.",
    )

    batch, states = logs.shape
    k = int(cardinality)

    require(
        batch > 0 and 1 <= k <= states,
        "Neispravna velicina skupa.",
    )
    require(
        np.allclose(
            logsumexp(logs, axis=1),
            0.0,
            atol=2e-9,
            rtol=0.0,
        ),
        "Ulazna registarska raspodela nije normalizovana.",
    )

    forward = np.full(
        (batch, states + 1, k + 1),
        -np.inf,
        dtype=np.float64,
    )
    backward = np.full_like(forward, -np.inf)

    forward[:, 0, 0] = 0.0

    for value in range(states):
        forward[:, value + 1, 0] = 0.0
        forward[:, value + 1, 1:] = np.logaddexp(
            forward[:, value, 1:],
            logs[:, value, None] + forward[:, value, :-1],
        )

    backward[:, states, 0] = 0.0

    for value in range(states - 1, -1, -1):
        backward[:, value, 0] = 0.0
        backward[:, value, 1:] = np.logaddexp(
            backward[:, value + 1, 1:],
            logs[:, value, None] + backward[:, value + 1, :-1],
        )

    log_partition = forward[:, states, k]

    if not np.isfinite(log_partition).all():
        raise DistributionError(
            "Raspodela nema masu na skupovima trazene velicine."
        )

    inclusion = np.empty((batch, states), dtype=np.float64)

    for value in range(states):
        excluding = logsumexp(
            forward[:, value, :k]
            + backward[:, value + 1, :k][:, ::-1],
            axis=1,
        )
        inclusion[:, value] = np.exp(
            logs[:, value] + excluding - log_partition
        )

    require(
        np.all(inclusion >= 0.0)
        and np.all(inclusion <= 1.0 + 2e-8)
        and np.allclose(
            inclusion.sum(axis=1),
            float(k),
            atol=2e-8,
            rtol=0.0,
        )
        and np.all(
            log_partition + math.lgamma(k + 1) <= 2e-8
        ),
        "Raspodela skupova nije pravilno normalizovana.",
    )

    losses = None

    if targets is not None:
        targets = np.asarray(targets, dtype=np.int64)

        require(
            targets.shape == (batch, k)
            and np.all((targets >= 0) & (targets < states))
            and np.all(np.diff(targets, axis=1) > 0),
            "Neispravni ciljni skupovi.",
        )

        target_logs = logs[
            np.arange(batch)[:, None], targets
        ].sum(axis=1)

        losses = log_partition - target_logs

        require(
            not np.isnan(losses).any()
            and np.all(losses >= -2e-8),
            "Neispravan NLL skupa.",
        )
        losses = np.maximum(losses, 0.0)

    return {
        "log_partition": log_partition,
        "inclusion": inclusion,
        "losses": losses,
    }


def map_set(log_probabilities, cardinality=K):
    """
    Tacan globalni MAP za ovu raspodelu skupova.

    Pri izjednacenim masama bira se manji bazni indeks.
    Sortiranje je samo prikaz neuredjenog skupa.
    """
    logs = np.asarray(log_probabilities, dtype=np.float64)
    require(logs.ndim == 1, "MAP ocekuje jedan vektor.")

    statistics = set_statistics(
        logs[None, :], cardinality=cardinality
    )

    order = np.lexsort((np.arange(len(logs)), -logs))
    chosen = np.sort(order[:cardinality])
    log_mass = float(logs[chosen].sum())
    log_partition = float(statistics["log_partition"][0])

    require(
        len(set(chosen.tolist())) == cardinality
        and np.isfinite(log_mass)
        and log_mass <= log_partition + 2e-8,
        "Neispravan MAP skupa.",
    )

    return {
        "path": chosen,
        "inclusion": statistics["inclusion"][0],
        "log_partition": log_partition,
        "log_probability": log_mass - log_partition,
    }


def objective(
    theta, features, targets, specification,
    with_gradient=True,
):
    require(
        len(features) == len(targets) and len(targets) > 0,
        "Neispravan trening skup.",
    )

    gradient = np.zeros(
        (SOURCE_QUBITS, features.shape[1]),
        dtype=np.float64,
    )
    loss_sum = 0.0

    for start in range(0, len(targets), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(targets))
        batch_features = features[start:stop]
        batch_targets = targets[start:stop]

        probabilities, cache = forward_probabilities(
            theta,
            batch_features,
            specification,
            record=with_gradient,
        )

        try:
            statistics = set_statistics(
                safe_log(probabilities),
                targets=batch_targets,
            )
        except DistributionError:
            return np.inf, np.zeros_like(theta)

        if not np.isfinite(statistics["losses"]).all():
            return np.inf, np.zeros_like(theta)

        loss_sum += float(statistics["losses"].sum())

        if not with_gradient:
            continue

        log_gradient = statistics["inclusion"].copy()
        log_gradient[
            np.arange(stop - start)[:, None],
            batch_targets,
        ] -= 1.0

        probability_gradient = np.zeros_like(probabilities)

        with np.errstate(
            divide="ignore", invalid="ignore", over="ignore"
        ):
            np.divide(
                log_gradient,
                probabilities,
                out=probability_gradient,
                where=probabilities > 0.0,
            )

        if not np.isfinite(probability_gradient).all():
            return np.inf, np.zeros_like(theta)

        source_gradient = probability_gradient[
            :, specification["lookup"]
        ]

        angle_gradient = np.empty(
            (stop - start, SOURCE_QUBITS),
            dtype=np.float64,
        )

        for bit in range(SOURCE_QUBITS):
            local_derivative = (
                0.5
                * np.sin(cache["angles"][:, bit])[:, None]
                * (2.0 * RAW_BITS[None, :, bit] - 1.0)
            )

            source_derivative = (
                cache["prefix"][:, :, bit]
                * cache["suffix"][:, :, bit + 1]
                * local_derivative
            )

            angle_gradient[:, bit] = np.sum(
                source_gradient * source_derivative,
                axis=1,
            )

        gradient += angle_gradient.T @ batch_features

    mask = regularization_mask(features.shape[1])
    denominator = max(1.0, float(mask.sum()))

    loss = (
        loss_sum / len(targets)
        + REGULARIZATION * np.dot(mask * theta, theta) / denominator
    )
    full_gradient = (
        gradient.ravel() / len(targets)
        + 2.0 * REGULARIZATION * mask * theta / denominator
    )

    require(
        np.isfinite(loss) and np.isfinite(full_gradient).all(),
        "Nekonacan gubitak ili gradijent.",
    )

    return float(loss), full_gradient


def fit_model(features, targets, specification):
    best = None

    for restart in range(RESTARTS):
        tracked = {
            "loss": np.inf,
            "theta": None,
            "calls": 0,
        }

        def tracked_objective(candidate):
            value, gradient = objective(
                candidate, features, targets, specification
            )
            tracked["calls"] += 1

            if np.isfinite(value) and value < tracked["loss"]:
                tracked["loss"] = value
                tracked["theta"] = candidate.copy()

            if DIAGNOSTICS and tracked["calls"] % 20 == 0:
                detail(
                    f"Pocetak {restart + 1}, "
                    f"procena {tracked['calls']}, "
                    f"NLL+L2={tracked['loss']:.9f}"
                )

            return value, gradient

        result = minimize(
            tracked_objective,
            initial_parameters(features.shape[1], restart),
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": MAX_ITERATIONS,
                "gtol": GRADIENT_TOLERANCE,
                "ftol": 1e-10,
                "maxls": 30,
                "maxcor": 10,
            },
        )

        require(
            tracked["theta"] is not None,
            "Optimizator nije nasao konacan model.",
        )

        chosen = tracked["theta"]
        loss, gradient = objective(
            chosen, features, targets, specification
        )
        gradient_norm = float(np.max(np.abs(gradient)))

        candidate = {
            "theta": chosen.copy(),
            "loss": loss,
            "gradient": gradient_norm,
            "converged": gradient_norm <= GRADIENT_TOLERANCE,
            "message": str(result.message),
        }

        if best is None or loss < best["loss"]:
            best = candidate

    return best


def evaluate(theta, features, targets, specification):
    losses = []
    hits = []

    for start in range(0, len(targets), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(targets))
        probabilities, _ = forward_probabilities(
            theta, features[start:stop], specification
        )
        logs = safe_log(probabilities)

        statistics = set_statistics(
            logs, targets=targets[start:stop]
        )
        losses.extend(statistics["losses"].tolist())

        for row, target in enumerate(targets[start:stop]):
            prediction = map_set(logs[row])["path"]
            hits.append(
                len(set(prediction.tolist()) & set(target.tolist()))
            )

    require(len(losses) > 0, "Prazna evaluacija.")
    losses = np.asarray(losses)

    return {
        "nll": float(losses.mean()),
        "hits": np.asarray(hits, dtype=np.int64),
        "zero_targets": int(np.isposinf(losses).sum()),
    }


def source_qiskit(theta, feature):
    weights = theta.reshape(SOURCE_QUBITS, len(feature))
    angles = weights @ feature

    circuit = QuantumCircuit(SOURCE_QUBITS)

    for bit in range(SOURCE_QUBITS):
        circuit.ry(float(angles[bit]), bit)

    for control in range(SOURCE_QUBITS - 1):
        circuit.cx(control, control + 1)

    actual = np.asarray(
        Statevector.from_instruction(circuit).data,
        dtype=np.complex128,
    )

    factors = np.where(
        RAW_BITS == 1,
        np.sin(angles / 2.0)[None, :],
        np.cos(angles / 2.0)[None, :],
    )
    raw_amplitudes = np.prod(factors, axis=1)
    expected = raw_amplitudes[SOURCE_INVERSE]

    require(
        np.allclose(
            actual, expected,
            atol=2e-10, rtol=2e-9,
        ),
        "Qiskit i numericko izvorno stanje se razlikuju.",
    )

    return actual


def append_oracle(circuit, specification, parameter):
    """
    Svaki izlazni bit racuna se kao H-D-H.

    D(x,b) = (-1)^(b * bit(f_r(x))).

    Zato je H-D-H kontrolisani X nad izlaznim bitom.
    Celo kolo izvodi y -> y XOR f_r(x).
    """
    for output_bit in range(OUTPUT_QUBITS):
        target = X_QUBITS + output_bit
        circuit.h(target)
        circuit.append(
            specification["phase_gates"][parameter][output_bit],
            list(range(X_QUBITS)) + [target],
        )
        circuit.h(target)


def simulate_oracle(
    branch, specification, parameter,
    output_seed=0, applications=1,
):
    require(
        branch.shape == (X_DIM,)
        and np.isclose(np.vdot(branch, branch).real, 1.0, atol=2e-10)
        and 0 <= parameter < R_DIM
        and 0 <= output_seed < OUTPUT_DIM,
        "Neispravan ulaz kvadratnog orakla.",
    )

    circuit = QuantumCircuit(X_QUBITS + OUTPUT_QUBITS)
    circuit.append(
        StatePreparation(branch), range(X_QUBITS)
    )

    for bit in range(OUTPUT_QUBITS):
        if (output_seed >> bit) & 1:
            circuit.x(X_QUBITS + bit)

    for _ in range(applications):
        append_oracle(circuit, specification, parameter)

    actual = np.asarray(
        Statevector.from_instruction(circuit).data,
        dtype=np.complex128,
    ).reshape(OUTPUT_DIM, X_DIM)

    expected = np.zeros_like(actual)
    outputs = np.full(X_DIM, output_seed, dtype=np.int64)

    if applications % 2:
        outputs ^= specification["table"][parameter]

    expected[outputs, np.arange(X_DIM)] = branch

    require(
        np.allclose(
            actual, expected,
            atol=2e-9, rtol=2e-9,
        ),
        "Qiskit orakl ne izvodi trazeno reverzibilno preslikavanje.",
    )

    return np.sum(np.abs(actual) ** 2, axis=1)


def qiskit_output(theta, feature, specification):
    """
    Tacan parcijalni trag po r registru preko njegovih blokova.
    Izlaz svakog bloka racuna stvarni Qiskit orakl.
    """
    source = source_qiskit(theta, feature).reshape(R_DIM, X_DIM)

    output = np.zeros(OUTPUT_DIM, dtype=np.float64)
    parameter_masses = np.zeros(R_DIM, dtype=np.float64)

    for parameter in range(R_DIM):
        branch = source[parameter]
        mass = float(np.vdot(branch, branch).real)
        parameter_masses[parameter] = mass

        if mass == 0.0:
            continue

        normalized_branch = branch / np.sqrt(mass)
        output += mass * simulate_oracle(
            normalized_branch,
            specification,
            parameter,
        )

    require(
        np.isclose(output.sum(), 1.0, atol=2e-9)
        and np.allclose(output[N:], 0.0, atol=2e-12),
        "Qiskit izlaz nije normalizovan u prostoru 1..39.",
    )

    numerical, _ = forward_probabilities(
        theta, feature[None, :], specification
    )

    require(
        np.allclose(
            output[:N], numerical[0],
            atol=2e-9, rtol=2e-8,
        ),
        "Qiskit i numericki prenos raspodele se razlikuju.",
    )

    return output[:N], parameter_masses


def self_checks(specifications):
    require(
        len(np.unique(SOURCE_PERMUTATION)) == SOURCE_DIM,
        "CNOT preslikavanje nije permutacija.",
    )

    # Algebarska konjugacija logisticke i kvadratne mape.
    for parameter in R_VALUES:
        x = X_VALUES.copy()
        z = parameter * (0.5 - x)
        c = parameter * (2.0 - parameter) / 4.0

        for _ in range(max(MAP_STEPS)):
            x = parameter * x * (1.0 - x)
            z = z * z + c

            require(
                np.allclose(
                    z, parameter * (0.5 - x),
                    atol=2e-12, rtol=2e-12,
                ),
                "Logisticka i Mandelbrotova koordinata nisu saglasne.",
            )

    branch = (
        np.sin(np.arange(X_DIM, dtype=float) + 0.3)
        + 1j * np.cos(np.arange(X_DIM, dtype=float) * 0.7)
    )
    branch /= np.linalg.norm(branch)

    for specification in specifications.values():
        require(
            np.all(specification["support"]),
            "Kandidat ne pokriva sve izlazne brojeve.",
        )

        simulate_oracle(
            branch, specification, 0, output_seed=37
        )
        simulate_oracle(
            branch, specification, R_DIM - 1, output_seed=7
        )
        simulate_oracle(
            branch, specification, 7,
            output_seed=19, applications=2,
        )

    # Potpuni pregled malog prostora skupova.
    toy = np.exp(
        0.4 * np.sin(np.arange(5, dtype=float))
    )
    toy /= toy.sum()
    logs = np.log(toy)
    statistics = set_statistics(logs[None, :], cardinality=3)
    prediction = map_set(logs, cardinality=3)

    combinations = list(itertools.combinations(range(5), 3))
    scores = np.array([
        logs[list(combination)].sum()
        for combination in combinations
    ])
    brute_inclusion = np.zeros(5)

    for combination, mass in zip(
        combinations, np.exp(scores - logsumexp(scores))
    ):
        brute_inclusion[list(combination)] += mass

    require(
        np.isclose(
            statistics["log_partition"][0],
            logsumexp(scores),
            atol=1e-12,
        )
        and np.allclose(
            statistics["inclusion"][0],
            brute_inclusion,
            atol=1e-12, rtol=0.0,
        )
        and np.isclose(
            logs[prediction["path"]].sum(),
            scores.max(),
            atol=1e-12,
        ),
        "Normalizator, ukljucenost ili MAP nisu tacni.",
    )

    sole = np.zeros(5)
    sole[[0, 2, 4]] = 1.0 / 3.0
    sole_statistics = set_statistics(
        safe_log(sole)[None, :], cardinality=3
    )
    sole_map = map_set(safe_log(sole), cardinality=3)

    require(
        np.array_equal(sole_map["path"], [0, 2, 4])
        and np.allclose(
            sole_statistics["inclusion"][0],
            [1.0, 0.0, 1.0, 0.0, 1.0],
            atol=1e-12,
        ),
        "Provera jedine moguce kombinacije nije prosla.",
    )

    impossible = np.array([0.5, 0.5, 0.0, 0.0, 0.0])

    try:
        set_statistics(
            safe_log(impossible)[None, :], cardinality=3
        )
    except DistributionError:
        pass
    else:
        raise RuntimeError("Nulta masa skupova nije prijavljena.")

    fixture = np.array([
        [0, 4, 9, 13, 20, 30, 38],
        [1, 6, 10, 14, 22, 31, 37],
        [0, 5, 12, 16, 24, 33, 38],
        [2, 7, 11, 18, 23, 29, 36],
        [1, 8, 14, 19, 26, 32, 37],
        [3, 9, 15, 21, 27, 34, 38],
    ], dtype=np.int64)

    times = np.arange(1, len(fixture))
    targets = fixture[1:]

    for window in WINDOWS:
        raw = raw_history_features(fixture, times, window)
        transform = fit_transform(raw, 3)
        features = transform_features(raw, transform)
        theta = initial_parameters(features.shape[1], 0)

        changed = fixture.copy()
        changed[4:] = changed[4:][::-1]
        changed_raw = raw_history_features(
            changed, times, window
        )
        changed_transform = fit_transform(changed_raw, 3)

        require(
            np.array_equal(raw[:4], changed_raw[:4])
            and np.array_equal(
                transform["keep"], changed_transform["keep"]
            )
            and np.allclose(
                transform["mean"], changed_transform["mean"]
            )
            and np.allclose(
                transform["scale"], changed_transform["scale"]
            ),
            "Buduci podaci uticu na ranije karakteristike.",
        )

        for specification in specifications.values():
            value, analytic = objective(
                theta, features, targets, specification
            )
            require(np.isfinite(value), "Nekonacan proverni gubitak.")

            selected = set(
                np.linspace(
                    0, len(theta) - 1, 18, dtype=int
                ).tolist()
            )

            for bit in range(SOURCE_QUBITS):
                selected.add(bit * features.shape[1])

            step = 1e-6

            for index in sorted(selected):
                upper = theta.copy()
                lower = theta.copy()
                upper[index] += step
                lower[index] -= step

                numeric = (
                    objective(
                        upper, features, targets,
                        specification, with_gradient=False,
                    )[0]
                    - objective(
                        lower, features, targets,
                        specification, with_gradient=False,
                    )[0]
                ) / (2.0 * step)

                require(
                    np.isclose(
                        analytic[index], numeric,
                        atol=2e-5, rtol=3e-4,
                    ),
                    "Koordinatna provera gradijenta nije prosla.",
                )

            for direction_index in range(3):
                direction = np.sin(
                    (np.arange(len(theta)) + 1.0)
                    * (0.31 + 0.17 * direction_index)
                )
                direction /= np.linalg.norm(direction)

                numeric = (
                    objective(
                        theta + step * direction,
                        features, targets, specification,
                        with_gradient=False,
                    )[0]
                    - objective(
                        theta - step * direction,
                        features, targets, specification,
                        with_gradient=False,
                    )[0]
                ) / (2.0 * step)

                require(
                    np.isclose(
                        analytic @ direction, numeric,
                        atol=2e-5, rtol=3e-4,
                    ),
                    "Usmerena provera gradijenta nije prosla.",
                )

            if window == WINDOWS[0]:
                qiskit_output(theta, features[0], specification)

    detail("Ugradjene algebra, orakl, gradijent, skup i hronologija provere su prosle.")


def longest_consecutive(values):
    longest = 1
    current = 1

    for difference in np.diff(values):
        current = current + 1 if difference == 1 else 1
        longest = max(longest, current)

    return longest


def process_dataset(dataset, specifications):
    data = dataset["data"]
    name = dataset["path"].name
    times = np.arange(1, len(data))
    targets = data[1:]
    total = len(targets)

    train_end = int(TRAIN_FRACTION * total)
    validation_end = int(VALIDATION_END_FRACTION * total)

    require(
        0 < train_end < validation_end < total,
        f"{name}: nedovoljno podataka za hronolosku podelu.",
    )

    print(f"{name} | obuka i izbor modela", flush=True)
    detail(f"Redova={len(data)}, SHA256={dataset['digest']}")

    best = None

    for window in WINDOWS:
        raw = raw_history_features(data, times, window)
        transform = fit_transform(raw, train_end)
        features = transform_features(raw, transform)

        for steps, specification in specifications.items():
            fit = fit_model(
                features[:train_end],
                targets[:train_end],
                specification,
            )

            try:
                metrics = evaluate(
                    fit["theta"],
                    features[train_end:validation_end],
                    targets[train_end:validation_end],
                    specification,
                )
            except DistributionError as exc:
                detail(f"Kandidat odbijen: {exc}")
                continue

            detail(
                f"Prozor={window}, iteracije={steps}, "
                f"validacioni NLL={metrics['nll']:.9f}"
            )

            if np.isfinite(metrics["nll"]) and (
                best is None or metrics["nll"] < best["metrics"]["nll"]
            ):
                best = {
                    "window": window,
                    "steps": steps,
                    "fit": fit,
                    "metrics": metrics,
                }

    require(best is not None, f"{name}: nema validnog modela.")

    window = best["window"]
    specification = specifications[best["steps"]]
    raw = raw_history_features(data, times, window)

    print(
        f"Validacija | prozor={window}, iteracije={best['steps']}, "
        f"NLL={best['metrics']['nll']:.6f}, "
        f"pogodaka={best['metrics']['hits'].mean():.4f}",
        flush=True,
    )

    test_transform = fit_transform(raw, validation_end)
    test_features = transform_features(raw, test_transform)

    test_fit = fit_model(
        test_features[:validation_end],
        targets[:validation_end],
        specification,
    )
    test_metrics = evaluate(
        test_fit["theta"],
        test_features[validation_end:],
        targets[validation_end:],
        specification,
    )

    print(
        f"Test | NLL={test_metrics['nll']:.6f}, "
        f"pogodaka={test_metrics['hits'].mean():.4f}",
        flush=True,
    )

    if test_metrics["zero_targets"]:
        print(
            f"Upozorenje: test ciljevi sa nultom masom="
            f"{test_metrics['zero_targets']}"
        )

    # Hiperparametri su izabrani pre testa.
    # Zavrsno ucenje obuhvata sve istorijske prelaze.
    final_transform = fit_transform(raw, len(raw))
    final_features = transform_features(raw, final_transform)

    final_fit = fit_model(
        final_features, targets, specification
    )

    next_raw = raw_history_features(
        data, np.array([len(data)]), window
    )
    next_features = transform_features(
        next_raw, final_transform
    )

    probabilities, parameter_masses = qiskit_output(
        final_fit["theta"],
        next_features[0],
        specification,
    )
    distribution = map_set(safe_log(probabilities))
    combination = distribution["path"] + 1

    require(
        len(combination) == K
        and len(set(combination.tolist())) == K
        and np.all((combination >= 1) & (combination <= N)),
        "Nelegalna NEXT kombinacija.",
    )

    unconverged = [
        label
        for label, fit in (
            ("validacioni model", best["fit"]),
            ("test model", test_fit),
            ("zavrsni model", final_fit),
        )
        if not fit["converged"]
    ]

    if unconverged:
        print(
            "Napomena: mali gradijent nije potvrdjen za "
            + ", ".join(unconverged)
            + "."
        )

    if np.array_equal(combination, data[-1] + 1):
        print("Upozorenje: NEXT ponavlja poslednje kolo.")

    run = longest_consecutive(combination)

    if run >= 5:
        print(f"Upozorenje: NEXT ima {run} uzastopnih brojeva.")

    span = int(combination[-1] - combination[0])
    span_percentile = float(
        np.mean((data[:, -1] - data[:, 0]) <= span)
    )

    if span_percentile <= 0.01:
        print("Upozorenje: izrazito grupisana NEXT kombinacija.")

    detail(
        "Log verovatnoce razlicitih izlaza: "
        f"{math.lgamma(K + 1) + distribution['log_partition']:.9f}"
    )
    detail(
        f"Log uslovne mase NEXT: {distribution['log_probability']:.9f}"
    )

    for parameter, mass in zip(R_VALUES, parameter_masses):
        detail(
            f"r={parameter:.12f}, "
            f"c={parameter * (2.0 - parameter) / 4.0:.12f}, "
            f"masa={mass:.9f}"
        )

    numbers = " ".join(str(int(value)) for value in combination)
    print(f"{name} NEXT: {numbers}", flush=True)

    return {
        "name": name,
        "digest": dataset["digest"],
        "next": combination,
        "probabilities": probabilities,
        "inclusion": distribution["inclusion"],
        "parameter_masses": parameter_masses,
    }


def main():
    print(
        "Quantum Logistic–Mandelbrot V1 | "
        "kvadratni prenos raspodele i neuredjeni skup od 7 brojeva"
    )
    print(
        f"Qiskit {qiskit.__version__} | "
        "6 ulaznih + 4 parametarska + 6 izlaznih kubita; "
        "blokovska simulacija"
    )

    specifications = {}

    for steps in MAP_STEPS:
        specification = build_specification(steps)

        if np.all(specification["support"]):
            specifications[steps] = specification
        else:
            print(
                f"Napomena: {steps} iteracija iskljuceno jer "
                "diskretizovani orakl ne pokriva svih 39 brojeva."
            )

    require(
        len(specifications) > 0,
        "Nijedna diskretizacija nema potpun izlazni prostor.",
    )

    self_checks(specifications)

    results = [
        process_dataset(load_csv(path), specifications)
        for path in FILES
    ]

    for first, second in itertools.combinations(results, 2):
        if first["digest"] == second["digest"]:
            print(
                f"Upozorenje: isti sadrzaj "
                f"{first['name']} i {second['name']}."
            )

        if np.array_equal(first["next"], second["next"]):
            probability_difference = float(np.max(np.abs(
                first["probabilities"] - second["probabilities"]
            )))
            inclusion_difference = float(np.max(np.abs(
                first["inclusion"] - second["inclusion"]
            )))
            parameter_difference = float(np.max(np.abs(
                first["parameter_masses"] - second["parameter_masses"]
            )))

            print(
                f"Isti NEXT: {first['name']} / {second['name']} | "
                f"max razlika p={probability_difference:.3e}, "
                f"ukljucenosti={inclusion_difference:.3e}, "
                f"r-mase={parameter_difference:.3e}"
            )


if __name__ == "__main__":
    main()



"""
Quantum Logistic-Mandelbrot V1 | kvadratni prenos raspodele i neuredjeni skup od 7 brojeva
Qiskit 1.4.4 | 6 ulaznih + 4 parametarska + 6 izlaznih kubita; blokovska simulacija

loto7_4716_k89.csv | obuka i izbor modela
Validacija | prozor=1, iteracije=1, NLL=16.878853, pogodaka=1.2744
Test | NLL=16.888811, pogodaka=1.2839
Napomena: mali gradijent nije potvrdjen za validacioni model, test model, zavrsni model.
loto7_4716_k89.csv NEXT: 2 3 9 10 13 31 34

loto7_4716_k89_loto_2980.csv | obuka i izbor modela
Validacija | prozor=1, iteracije=1, NLL=16.915659, pogodaka=1.3087
Test | NLL=16.957696, pogodaka=1.2707
Napomena: mali gradijent nije potvrdjen za validacioni model, test model, zavrsni model.
loto7_4716_k89_loto_2980.csv NEXT: 2 3 14 15 17 34 35

loto7_4716_k89_loto_plus_1736.csv | obuka i izbor modela
Validacija | prozor=1, iteracije=1, NLL=17.081420, pogodaka=1.2269
Test | NLL=16.872182, pogodaka=1.3985
Napomena: mali gradijent nije potvrdjen za validacioni model, test model, zavrsni model.
loto7_4716_k89_loto_plus_1736.csv NEXT: 8 10 25 27 28 30 31
"""



"""
Prenos raspodele kroz kvadratnu dinamiku.

Povezana predstavljanja iste dinamike. Logistička dinamika i bifurkacije

- Učenje parametra kvadratne mape iz istorijskih prelaza.
- Prenos cele raspodele kroz mapu, uključujući njeno presavijanje: različita početna stanja mogu doprineti istom izlazu.
- Bifurkaciona struktura kao deo pretrage parametara i periodičnih režima.

Posebnost predikcije bila bi upravo taj nelinearni prenos raspodele. 
Mandelbrotova slika služila bi za organizovanje prostora parametara.

Qiskit izvedba zahteva reverzibilni orakl sa pomoćnim registrom. 
Logistička mapa je više-na-jedan, pa sama nije unitarna kapija. 
Orakl bi zadržao ulaz i izračunao diskretizovani izlaz; 
izlazni registar zatim daje prenetu raspodelu. 
Time osnovna mapa stvarno određuje predikciju.

Osnova je kvadratna mapa i njen reverzibilni Qiskit orakl. 
Model koristi kvadratnu logističku dinamiku, reverzibilni Qiskit orakl i raspodelu neuređenih skupova od sedam brojeva.
"""
