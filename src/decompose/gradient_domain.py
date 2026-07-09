"""
src/decompose/gradient_domain.py
Atenuacion continua multiescala en dominio de gradiente, generalizando
Retinex-Horn (umbral binario) con pesos continuos por escala. Linea Fattal,
Lischinski y Werman (2002), extendida por Milovic, Conejero y Tejos (2016),
ecuacion 4: Q_k = ||grad R_k||^(beta_k - 1) sobre niveles residuales de una
piramide multiescala.

La magnitud de los coeficientes de detalle de la regla clasica no discrimina
bien reflectancia de shading contra el ground-truth, pero existe margen real
para una regla mejor. Este modulo no umbraliza: construye un peso continuo
p(x) en [0, 1] por pixel a partir de la estructura del gradiente a traves de
escalas, en vez de un corte binario por magnitud con MAD.

Diferencias deliberadas respecto del paper:
- Se usa cromaticidad |grad C| en vez de luminancia como factor multiplicativo
  opcional del peso (nuestra senal fisica es la croma, no el SNR astronomico).
- El paper usa W como mapa de atenuacion de un gradiente unico (compresion para
  visualizacion). Aqui se reinterpreta como peso de mezcla p(x) entre dos
  reconstrucciones (reflectancia y shading) que deben sumar el gradiente
  original, para lo cual el producto de Q_k (no acotado) se satura a [0, 1]
  con p = raw / (1 + raw). beta_k = 1 en todas las escalas da raw = 1 en todas
  partes, es decir p = 0.5 (reparto neutro), consistente con "Q_k = 1" siendo
  el punto neutro del paper (ni atenua ni amplifica).

Generalizacion de transformada: la ecuacion 4 solo necesita una piramide de
aproximaciones R_k sucesivamente mas suaves, no es especifica de la starlet.
Se agregan piramides equivalentes para MMT (mediana a trous) y Wavelet SWT
db2, seleccionables con transform_type.
"""

import numpy as np
import pywt
from scipy.ndimage import convolve1d, median_filter
from typing import Tuple

from src.baselines.homomorphic import rgb_to_luminance
from src.transforms.starlet import get_starlet_filter
from src.transforms.mmt import atrous_median_filter2d
from src.decompose.multiscale import median_absolute_deviation
from src.utils.poisson import get_gradients, solve


def starlet_residual_pyramid(image: np.ndarray, levels: int) -> list:
    """
    Calcula la piramide de residuos R_k de la starlet (a trous), sin exponer
    los coeficientes de detalle. R_0 es la imagen de entrada, R_k es la
    aproximacion sucesivamente suavizada tras k iteraciones del filtro
    B3-spline, y R_levels coincide con el residuo final que retorna
    starlet_decompose.

    Args:
        image (np.ndarray): Imagen 2D [H, W].
        levels (int): Numero de niveles L.

    Returns:
        list: [R_0, R_1, ..., R_levels], longitud levels + 1.
    """
    pyramid = [image.astype(np.float64)]
    c_j = pyramid[0]
    for j in range(levels):
        h = get_starlet_filter(j)
        c_next = convolve1d(c_j, h, axis=0, mode='reflect')
        c_next = convolve1d(c_next, h, axis=1, mode='reflect')
        pyramid.append(c_next)
        c_j = c_next
    return pyramid


def mmt_residual_pyramid(image: np.ndarray, levels: int, kernel_size: int = 3) -> list:
    """
    Piramide de residuos R_k de la MMT (mediana a trous), analoga a
    starlet_residual_pyramid pero reemplazando la convolucion B3-spline por el
    filtro de mediana movil de src.transforms.mmt.

    Args:
        image (np.ndarray): Imagen 2D [H, W].
        levels (int): Numero de niveles L.
        kernel_size (int): Tamano de la ventana de mediana (3 o 5).

    Returns:
        list: [R_0, R_1, ..., R_levels], longitud levels + 1.
    """
    pyramid = [image.astype(np.float64)]
    c_j = pyramid[0]
    for j in range(levels):
        step = 2 ** j
        c_next = atrous_median_filter2d(c_j, step, kernel_size)
        pyramid.append(c_next)
        c_j = c_next
    return pyramid


def wavelet_residual_pyramid(image: np.ndarray, levels: int, wavelet_name: str = "db2") -> list:
    """
    Piramide de residuos R_k de la SWT (a trous, wavelets db2), analoga a
    starlet_residual_pyramid. Las aproximaciones cA_j que entrega pywt.swt2 no
    preservan amplitud (crecen en un factor 2 por nivel, normalizacion
    ortogonal estandar de la libreria): se renormalizan dividiendo por
    2^profundidad para que R_k quede en las mismas unidades que la imagen de
    entrada, igual que en starlet y MMT.

    Args:
        image (np.ndarray): Imagen 2D [H, W].
        levels (int): Numero de niveles L.
        wavelet_name (str): Familia de wavelet (por defecto "db2").

    Returns:
        list: [R_0, R_1, ..., R_levels], longitud levels + 1.
    """
    H, W = image.shape
    factor = 2 ** levels
    pad_H = ((H + factor - 1) // factor) * factor
    pad_W = ((W + factor - 1) // factor) * factor
    padded = np.pad(image, ((0, pad_H - H), (0, pad_W - W)), mode='reflect')

    coeffs = pywt.swt2(padded, wavelet_name, level=levels)

    pyramid = [image.astype(np.float64)]
    for depth in range(1, levels + 1):
        c_a = coeffs[levels - depth][0]
        r_k = (c_a / (2.0 ** depth))[:H, :W]
        pyramid.append(r_k)
    return pyramid


def residual_pyramid(image: np.ndarray, levels: int, transform_type: str = "starlet",
                      median_kernel_size: int = 3, wavelet_name: str = "db2") -> list:
    """
    Selecciona la piramide de residuos segun la transformada. La ecuacion 4 de
    Milovic et al. (2016) solo necesita una secuencia de aproximaciones R_k
    sucesivamente mas suaves, sin importar si esas aproximaciones vienen de un
    filtro lineal (starlet), un filtro de mediana (MMT) o una wavelet (SWT).

    Args:
        image (np.ndarray): Imagen 2D [H, W].
        levels (int): Numero de niveles L.
        transform_type (str): "starlet", "mmt" o "wavelet".
        median_kernel_size (int): Ventana de mediana, solo si transform_type="mmt".
        wavelet_name (str): Familia de wavelet, solo si transform_type="wavelet".

    Returns:
        list: [R_0, R_1, ..., R_levels], longitud levels + 1.
    """
    if transform_type == "starlet":
        return starlet_residual_pyramid(image, levels)
    elif transform_type == "mmt":
        return mmt_residual_pyramid(image, levels, median_kernel_size)
    elif transform_type == "wavelet":
        return wavelet_residual_pyramid(image, levels, wavelet_name)
    else:
        raise ValueError(f"transform_type '{transform_type}' no soportado")


def centered_gradient_magnitude(field: np.ndarray) -> np.ndarray:
    """
    Norma del gradiente por diferencias centradas (ecuacion 4 de Milovic et al.
    2016: "gradient components calculated by non-interlaced central
    differences"). Los bordes usan diferencia hacia adentro de un solo lado.

    Args:
        field (np.ndarray): Campo escalar 2D [H, W].

    Returns:
        np.ndarray: ||grad field||_2, forma [H, W].
    """
    grad_x = np.zeros_like(field)
    grad_y = np.zeros_like(field)
    grad_x[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / 2.0
    grad_x[:, 0] = field[:, 1] - field[:, 0]
    grad_x[:, -1] = field[:, -1] - field[:, -2]
    grad_y[1:-1, :] = (field[2:, :] - field[:-2, :]) / 2.0
    grad_y[0, :] = field[1, :] - field[0, :]
    grad_y[-1, :] = field[-1, :] - field[-2, :]
    return np.sqrt(grad_x ** 2 + grad_y ** 2)


def beta_values(levels: int, beta_fine: float, beta_coarse: float,
                 beta_profile: str = "linear") -> np.ndarray:
    """
    Perfil de beta por escala (indice 0 = mas fina, levels-1 = mas gruesa).

    Args:
        levels (int): Numero de escalas.
        beta_fine (float): Valor de beta en la escala mas fina.
        beta_coarse (float): Valor de beta en la escala mas gruesa (ignorado si
            beta_profile == "constant").
        beta_profile (str): "linear" (interpolacion lineal, como en el paper) o
            "constant" (beta_fine en todas las escalas).

    Returns:
        np.ndarray: Vector de longitud levels con beta_k.
    """
    if beta_profile == "constant":
        return np.full(levels, beta_fine, dtype=np.float64)
    elif beta_profile == "linear":
        if levels == 1:
            return np.array([beta_fine], dtype=np.float64)
        return np.linspace(beta_fine, beta_coarse, levels)
    else:
        raise ValueError(f"beta_profile '{beta_profile}' no soportado")


def compute_reflectance_weight(log_L_filled: np.ndarray, mask: np.ndarray, levels: int,
                                beta_fine: float = 1.1, beta_coarse: float = 0.1,
                                beta_profile: str = "linear",
                                chroma_map: np.ndarray = None,
                                chroma_gamma: float = 1.0,
                                color_threshold: float = 0.05,
                                median_smooth: bool = True,
                                transform_type: str = "starlet",
                                median_kernel_size: int = 3,
                                wavelet_name: str = "db2") -> np.ndarray:
    """
    Construye el peso continuo de reflectancia p(x) en [0, 1] a partir de la
    piramide de residuos de la transformada elegida (ecuacion 4 de Milovic
    et al. 2016).

    Args:
        log_L_filled (np.ndarray): Log-luminancia [H, W], fondo ya relleno.
        mask (np.ndarray): Mascara booleana [H, W] del foreground.
        levels (int): Numero de escalas de la piramide.
        beta_fine (float): Beta en la escala mas fina.
        beta_coarse (float): Beta en la escala mas gruesa.
        beta_profile (str): "linear" o "constant".
        chroma_map (np.ndarray, optional): |grad C| precalculado [H, W]. Si se
            entrega, modula el peso como factor multiplicativo adicional
            (sustituye a la luminancia del paper por la cromaticidad, nuestra
            senal fisica de shading acromatico).
        chroma_gamma (float): Exponente de la modulacion de croma.
        color_threshold (float): Multiplicador de la MAD de croma para T_c.
        median_smooth (bool): Si aplica suavizado mediana 3x3 al peso final
            (Apendice de Milovic et al. 2016), antes de usarlo en el gradiente.
        transform_type (str): "starlet", "mmt" o "wavelet".
        median_kernel_size (int): Ventana de mediana, solo si transform_type="mmt".
        wavelet_name (str): Familia de wavelet, solo si transform_type="wavelet".

    Returns:
        np.ndarray: Peso de reflectancia p(x) en [0, 1], forma [H, W].
    """
    pyramid = residual_pyramid(log_L_filled, levels, transform_type, median_kernel_size, wavelet_name)
    betas = beta_values(levels, beta_fine, beta_coarse, beta_profile)

    raw = np.ones_like(log_L_filled)
    for j in range(levels):
        R_k = pyramid[j + 1]
        grad_mag = centered_gradient_magnitude(R_k)
        grad_mag = np.maximum(grad_mag, 1e-8)
        Q_k = grad_mag ** (betas[j] - 1.0)
        raw *= Q_k

    if chroma_map is not None:
        T_c = color_threshold * median_absolute_deviation(chroma_map, mask)
        T_c = max(T_c, 1e-6)
        raw *= (chroma_map / T_c) ** chroma_gamma

    p = raw / (1.0 + raw)

    if median_smooth:
        p = median_filter(p, size=3)

    return np.clip(p, 0.0, 1.0)


def reintegrate_from_weight(log_L_filled: np.ndarray, mask: np.ndarray,
                             p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reintegra log-reflectancia y log-shading resolviendo dos ecuaciones de
    Poisson independientes sobre el gradiente de la log-luminancia ponderado
    por p(x) y (1 - p(x)) respectivamente, con el mismo solver que usa Horn.

    Args:
        log_L_filled (np.ndarray): Log-luminancia [H, W], fondo ya relleno.
        mask (np.ndarray): Mascara booleana [H, W] del foreground.
        p (np.ndarray): Peso de reflectancia en [0, 1], forma [H, W].

    Returns:
        Tuple[np.ndarray, np.ndarray]: log_R_L, log_S_L (ambos [H, W]).
    """
    i_y, i_x = get_gradients(log_L_filled)
    log_R_L = solve(p * i_y, p * i_x, mask)
    log_S_L = solve((1.0 - p) * i_y, (1.0 - p) * i_x, mask)
    return log_R_L, log_S_L


def gradient_domain_decomposition(image: np.ndarray, mask: np.ndarray, levels: int = 3,
                                   beta_fine: float = 1.1, beta_coarse: float = 0.1,
                                   beta_profile: str = "linear",
                                   chroma_modulation: bool = False,
                                   chroma_gamma: float = 1.0,
                                   color_threshold: float = 0.05,
                                   median_smooth: bool = True,
                                   transform_type: str = "starlet",
                                   median_kernel_size: int = 3,
                                   wavelet_name: str = "db2",
                                   eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Descomposicion R/S por atenuacion continua multiescala en dominio de
    gradiente ("Horn multiescala con pesos continuos"). Generaliza Retinex-Horn
    reemplazando su umbral binario de
    gradiente por un peso continuo p(x) construido con la ecuacion 4 de
    Milovic et al. (2016) sobre niveles residuales de una piramide multiescala,
    seleccionable entre starlet, MMT o wavelet (transform_type).

    Args:
        image (np.ndarray): Imagen de entrada [H, W] o [H, W, 3], rango [0, 1].
        mask (np.ndarray): Mascara booleana [H, W] del foreground.
        levels (int): Numero de escalas de la piramide usada para p(x).
        beta_fine (float): Beta en la escala mas fina.
        beta_coarse (float): Beta en la escala mas gruesa.
        beta_profile (str): "linear" o "constant".
        chroma_modulation (bool): Si module p(x) con el gradiente de
            cromaticidad (solo tiene efecto si la imagen es color).
        chroma_gamma (float): Exponente de la modulacion de croma.
        color_threshold (float): Multiplicador de la MAD de croma para T_c.
        median_smooth (bool): Si suaviza p(x) con mediana 3x3 antes de usarlo.
        transform_type (str): "starlet", "mmt" o "wavelet".
        median_kernel_size (int): Ventana de mediana, solo si transform_type="mmt".
        wavelet_name (str): Familia de wavelet, solo si transform_type="wavelet".
        eps (float): Constante para evitar log(0) y division por cero.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Shading estimado S [H, W].
            - Reflectancia estimada R [H, W] o [H, W, 3].
    """
    is_color = (image.ndim == 3)

    if is_color:
        L = rgb_to_luminance(image)
    else:
        L = image.copy()

    L_masked = np.clip(L, eps, 1.0)
    log_L = np.log(L_masked)
    if not np.all(mask):
        fg_mean = np.mean(log_L[mask]) if np.any(mask) else 0.0
        log_L_filled = np.where(mask, log_L, fg_mean)
    else:
        log_L_filled = log_L

    chroma_map = None
    if is_color and chroma_modulation:
        C = image / (L[..., np.newaxis] + eps)
        grad_C_x = np.zeros_like(C)
        grad_C_y = np.zeros_like(C)
        grad_C_x[:, :-1, :] = C[:, 1:, :] - C[:, :-1, :]
        grad_C_y[:-1, :, :] = C[1:, :, :] - C[:-1, :, :]
        chroma_map = np.sqrt(np.sum(grad_C_x ** 2 + grad_C_y ** 2, axis=-1))

    p = compute_reflectance_weight(
        log_L_filled, mask, levels, beta_fine, beta_coarse, beta_profile,
        chroma_map, chroma_gamma, color_threshold, median_smooth,
        transform_type, median_kernel_size, wavelet_name,
    )

    log_R_L, log_S_L = reintegrate_from_weight(log_L_filled, mask, p)

    R_L = np.clip(np.exp(log_R_L), 0.0, 1.0)
    S = np.clip(np.exp(log_S_L), 0.0, 1.0)

    if is_color:
        C = image / (L[..., np.newaxis] + eps)
        R = C * R_L[..., np.newaxis]
    else:
        R = R_L

    S = S * mask
    if is_color:
        R = R * mask[..., np.newaxis]
    else:
        R = R * mask

    return S, R
