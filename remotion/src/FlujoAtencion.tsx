/**
 * FlujoAtencion — Sesión 3
 *
 * Qué es: animación del pipeline de scaled dot-product attention.
 * Qué hace: muestra las 4 etapas en secuencia sobre matrices 4×4 reales
 * (calculadas con una semilla fija):
 *   1. scores = QKᵀ         (compatibilidad query–key)
 *   2. ÷ √d_k               (estabilizar el softmax)
 *   3. + máscara causal      (el futuro se apaga)
 *   4. softmax               (cada fila se vuelve una distribución)
 *
 * Los números son REALES: el mismo cálculo del Lab 05, frame a frame.
 */
import {AbsoluteFill, interpolate, useCurrentFrame} from 'remotion';

const TINTA = '#1a2332';
const TOKENS = ['El', 'gato', 'come', 'pescado'];
const D_K = 8;

// ── Generador determinístico (mulberry32): mismos números en cada render ──
function mulberry32(semilla: number) {
  return () => {
    semilla |= 0; semilla = (semilla + 0x6d2b79f5) | 0;
    let t = Math.imul(semilla ^ (semilla >>> 15), 1 | semilla);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ── Calcular el pipeline completo una sola vez ───────────────────────
const rand = mulberry32(42);
const gauss = () => (rand() + rand() + rand() - 1.5) * 1.4;
const Q = Array.from({length: 4}, () => Array.from({length: D_K}, gauss));
const K = Array.from({length: 4}, () => Array.from({length: D_K}, gauss));

const scores = Q.map((q) => K.map((k) => q.reduce((s, v, i) => s + v * k[i], 0)));
const escalado = scores.map((f) => f.map((v) => v / Math.sqrt(D_K)));
const enmascarado = escalado.map((f, i) => f.map((v, j) => (j > i ? -Infinity : v)));
const pesos = enmascarado.map((fila) => {
  const m = Math.max(...fila.filter((v) => isFinite(v)));
  const e = fila.map((v) => (isFinite(v) ? Math.exp(v - m) : 0));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
});

// Las 4 etapas de la animación, con su matriz y descripción
const ETAPAS = [
  {titulo: '1 · scores = QKᵀ', sub: 'compatibilidad entre cada query y cada key', M: scores, esPeso: false},
  {titulo: '2 · scores ÷ √d_k', sub: 'sin escalar, el softmax se satura al crecer d_k', M: escalado, esPeso: false},
  {titulo: '3 · + máscara causal', sub: 'el futuro se apaga: −∞ antes del softmax', M: enmascarado, esPeso: false},
  {titulo: '4 · softmax por filas', sub: 'cada token reparte atención que suma 1', M: pesos, esPeso: true},
];

/** Color de celda: divergente para scores, azul de intensidad para pesos. */
function colorCelda(v: number, vmax: number, esPeso: boolean): string {
  if (!isFinite(v)) return '#f1f5f9';
  if (esPeso) return `rgba(0, 114, 178, ${0.08 + 0.92 * Math.min(1, v)})`;
  const t = Math.max(-1, Math.min(1, v / vmax));
  return t >= 0
    ? `rgba(0, 158, 115, ${0.1 + 0.75 * t})`
    : `rgba(213, 94, 0, ${0.1 - 0.75 * t})`;
}

export const FlujoAtencion: React.FC = () => {
  const frame = useCurrentFrame();
  // Cada etapa dura 100 frames; la última se queda hasta el final
  const etapaIdx = Math.min(Math.floor(frame / 100), ETAPAS.length - 1);
  const etapa = ETAPAS[etapaIdx];
  // Fundido de entrada de cada etapa
  const opacidad = interpolate(frame - etapaIdx * 100, [0, 18], [0, 1], {
    extrapolateLeft: 'clamp', extrapolateRight: 'clamp',
  });

  const vmax = Math.max(...etapa.M.flat().filter(isFinite).map(Math.abs), 1e-9);
  const CELDA = 150;
  const X0 = 620, Y0 = 330;

  return (
    <AbsoluteFill style={{backgroundColor: '#f6f8fb', fontFamily: 'sans-serif'}}>
      <svg width="1920" height="1080">
        <text x={960} y={100} textAnchor="middle" fontSize={50} fontWeight="bold" fill={TINTA}>
          Attention(Q, K, V) = softmax(QKᵀ / √d_k + M) · V
        </text>

        {/* Barra de progreso de etapas */}
        {ETAPAS.map((e, i) => (
          <g key={i}>
            <rect x={330 + i * 330} y={150} width={310} height={56} rx={12}
              fill={i === etapaIdx ? '#0072B2' : i < etapaIdx ? '#9dc3e6' : '#e2e8f0'} />
            <text x={330 + i * 330 + 155} y={186} textAnchor="middle" fontSize={24}
              fontWeight="bold" fill={i <= etapaIdx ? '#fff' : '#94a3b8'}>
              {e.titulo}
            </text>
          </g>
        ))}

        <g opacity={opacidad}>
          <text x={960} y={280} textAnchor="middle" fontSize={34} fontStyle="italic" fill="#475569">
            {etapa.sub}
          </text>

          {/* Etiquetas de tokens (columnas = keys, filas = queries) */}
          {TOKENS.map((t, j) => (
            <text key={`c${j}`} x={X0 + j * CELDA + CELDA / 2} y={Y0 - 18}
              textAnchor="middle" fontSize={30} fontWeight="bold" fill={TINTA}>{t}</text>
          ))}
          {TOKENS.map((t, i) => (
            <text key={`r${i}`} x={X0 - 24} y={Y0 + i * CELDA + CELDA / 2 + 10}
              textAnchor="end" fontSize={30} fontWeight="bold" fill={TINTA}>{t}</text>
          ))}

          {/* La matriz de la etapa actual */}
          {etapa.M.map((fila, i) =>
            fila.map((v, j) => (
              <g key={`${i}-${j}`}>
                <rect x={X0 + j * CELDA} y={Y0 + i * CELDA} width={CELDA - 6} height={CELDA - 6}
                  rx={10} fill={colorCelda(v, vmax, etapa.esPeso)}
                  stroke="#cbd5e1" strokeWidth={1.5} />
                <text x={X0 + j * CELDA + (CELDA - 6) / 2} y={Y0 + i * CELDA + CELDA / 2 + 8}
                  textAnchor="middle" fontSize={30} fontFamily="monospace" fontWeight="bold"
                  fill={etapa.esPeso && v > 0.45 ? '#fff' : TINTA}>
                  {isFinite(v) ? v.toFixed(2) : '−∞'}
                </text>
              </g>
            ))
          )}

          {/* En la etapa softmax: verificación de que cada fila suma 1 */}
          {etapa.esPeso &&
            pesos.map((fila, i) => (
              <text key={`s${i}`} x={X0 + 4 * CELDA + 40} y={Y0 + i * CELDA + CELDA / 2 + 8}
                fontSize={28} fontFamily="monospace" fill="#009E73" fontWeight="bold">
                Σ = {fila.reduce((a, b) => a + b, 0).toFixed(2)} ✓
              </text>
            ))}
        </g>

        <text x={960} y={1020} textAnchor="middle" fontSize={30} fill="#64748b">
          El paso final (no mostrado): salida = pesos × V — la mezcla ponderada de los values.
        </text>
      </svg>
    </AbsoluteFill>
  );
};
