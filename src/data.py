"""Preparación de datos de los laboratorios.

Qué es: módulo de carga y preprocesamiento de datasets.
Qué hace: construye los DataLoaders de los tres laboratorios
(make_moons, FashionMNIST y texto tokenizado simple), aplicando las
reglas anti-leakage del curso: las estadísticas de normalización y el
vocabulario se ajustan SOLO con el train set.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

# ────────────────────────────────────────────────────────────────────
# Laboratorio 1 — make_moons (clasificación binaria no lineal)
# ────────────────────────────────────────────────────────────────────


def cargar_moons(
    n_samples: int = 1500,
    noise: float = 0.22,
    batch_size: int = 64,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, object]:
    """Genera make_moons y devuelve (train, val, test) DataLoaders.

    Decisiones importantes:
    - split estratificado 70/15/15 para conservar el balance de clases;
    - StandardScaler ajustado SOLO con train (evita data leakage);
    - labels con shape (N, 1) en float32 porque BCEWithLogitsLoss
      espera ese formato.
    """
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)

    # 70% train / 30% temporal, luego el temporal se parte 50/50
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed
    )

    # El scaler aprende media/desviación SOLO del train set:
    # validation y test se transforman con esas mismas estadísticas.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    def hacer_loader(X_: np.ndarray, y_: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(
            torch.tensor(X_, dtype=torch.float32),
            torch.tensor(y_, dtype=torch.float32).unsqueeze(1),  # (N,) → (N,1)
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return (
        hacer_loader(X_train, y_train, shuffle=True),   # solo train se baraja
        hacer_loader(X_val, y_val, shuffle=False),
        hacer_loader(X_test, y_test, shuffle=False),
        scaler,
    )


# ────────────────────────────────────────────────────────────────────
# Laboratorio 2 — FashionMNIST (visión, 10 clases)
# ────────────────────────────────────────────────────────────────────


def cargar_fashionmnist(
    root: str = "data",
    train_size: int = 54_000,
    batch_size: int = 128,
    eval_batch_size: int = 256,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Devuelve (train, val, test) DataLoaders de FashionMNIST.

    Detalle sutil pero crucial: train y validation usan TRANSFORMS
    DIFERENTES (train con augmentation, validation sin ella), por lo
    que se cargan dos copias del dataset y se comparten los índices
    del split. Evaluar con augmentation distorsionaría las métricas.
    """
    from torchvision import datasets, transforms

    # Augmentation SOLO para entrenamiento: transformaciones plausibles
    # que preservan la etiqueta (una camiseta girada sigue siendo camiseta).
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        # Estadísticas del canal calculadas sobre el train set
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    full_aug = datasets.FashionMNIST(root, train=True, download=True,
                                     transform=transform_train)
    full_eval = datasets.FashionMNIST(root, train=True, download=True,
                                      transform=transform_eval)
    test = datasets.FashionMNIST(root, train=False, download=True,
                                 transform=transform_eval)

    # Split determinístico de índices compartido entre ambas copias
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(full_aug), generator=generator).tolist()
    idx_train, idx_val = indices[:train_size], indices[train_size:]

    train_ds = torch.utils.data.Subset(full_aug, idx_train)
    val_ds = torch.utils.data.Subset(full_eval, idx_val)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False),
        DataLoader(test, batch_size=eval_batch_size, shuffle=False),
        test.classes,  # nombres de las 10 clases
    )


# ────────────────────────────────────────────────────────────────────
# Proyecto final — texto con vocabulario propio
# ────────────────────────────────────────────────────────────────────


def tokenizar_basico(texto: str) -> list[str]:
    """Tokenizador mínimo por palabras y puntuación (en minúsculas).

    Es deliberadamente simple: sirve para ENTENDER qué hace un
    tokenizer antes de usar los subword tokenizers de Hugging Face.
    """
    return re.findall(r"[A-Za-zÁÉÍÓÚáéíóúñÑ']+|[.,!?;]", texto.lower())


def construir_vocabulario(
    textos: list[str], min_frecuencia: int = 2
) -> tuple[list[str], dict[str, int]]:
    """Construye el vocabulario a partir del TRAIN set únicamente.

    Devuelve (itos, stoi): lista índice→token y diccionario token→índice.
    Los tokens especiales <pad> y <unk> ocupan las posiciones 0 y 1.
    """
    contador: Counter[str] = Counter()
    for texto in textos:
        contador.update(tokenizar_basico(texto))

    itos = ["<pad>", "<unk>"] + [
        token for token, freq in contador.items() if freq >= min_frecuencia
    ]
    stoi = {token: i for i, token in enumerate(itos)}
    return itos, stoi


class TextoDataset(Dataset):
    """Dataset de texto → IDs de vocabulario, con truncation.

    Cada muestra es (tensor de IDs de longitud variable, label).
    El padding se hace por batch en `collate_texto` (dynamic padding):
    rellenar al máximo del BATCH y no del dataset ahorra cómputo.
    """

    def __init__(self, textos: list[str], labels: list[int],
                 stoi: dict[str, int], max_length: int = 128) -> None:
        self.textos = textos
        self.labels = labels
        self.stoi = stoi
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, indice: int) -> tuple[torch.Tensor, int]:
        tokens = tokenizar_basico(self.textos[indice])[: self.max_length]
        ids = [self.stoi.get(token, 1) for token in tokens]  # 1 = <unk>
        return torch.tensor(ids, dtype=torch.long), int(self.labels[indice])


def collate_texto(batch: list[tuple[torch.Tensor, int]]):
    """Collate function con dynamic padding y máscara.

    Devuelve:
    - padded: (B, T_max_del_batch) con 0 (<pad>) en las posiciones vacías;
    - mask:   (B, T) booleana, True = token real, False = padding;
    - labels: (B,) enteros.
    La máscara evita que el padding contamine el pooling o la attention.
    """
    secuencias, labels = zip(*batch)
    max_len = max(len(s) for s in secuencias)

    padded = torch.zeros((len(secuencias), max_len), dtype=torch.long)
    mask = torch.zeros((len(secuencias), max_len), dtype=torch.bool)
    for fila, secuencia in enumerate(secuencias):
        padded[fila, : len(secuencia)] = secuencia
        mask[fila, : len(secuencia)] = True

    return padded, mask, torch.tensor(labels, dtype=torch.long)
