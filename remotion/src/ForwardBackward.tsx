/**
 * ForwardBackward — Sesión 1
 *
 * Qué es: animación del grafo computacional de L = (wx + b − y)².
 * Qué hace: en la primera mitad, los VALORES fluyen hacia adelante
 * (forward pass, azul); en la segunda, los GRADIENTES fluyen hacia
 * atrás (backward pass, rojo), mostrando la regla de la cadena como
 * producto de derivadas locales.
 *
 * Valores del ejemplo (los mismos del Lab 01):
 *   x=2, w=3, b=1, y=10  →  ŷ=7, L=9, dL/dw=−12, dL/db=−6
 */
import {AbsoluteFill, interpolate, useCurrentFrame} from 'remotion';

// ── Paleta del curso (Okabe-Ito) ─────────────────────────────────────
const AZUL = '#0072B2';
const ROJO = '#D55E00';
const NARANJA = '#E69F00';
const TINTA = '#1a2332';

// Posiciones fijas de los nodos del grafo (coordenadas del lienzo)
const NODOS = {
  x: {cx: 260, cy: 380, texto: 'x = 2', tipo: 'dato'},
  w: {cx: 260, cy: 620, texto: 'w = 3', tipo: 'param'},
  mul: {cx: 620, cy: 500, texto: '×', tipo: 'op'},
  b: {cx: 620, cy: 780, texto: 'b = 1', tipo: 'param'},
  suma: {cx: 950, cy: 620, texto: '+', tipo: 'op'},
  resta: {cx: 1280, cy: 620, texto: '− y', tipo: 'op'},
  cuad: {cx: 1570, cy: 620, texto: '(·)²', tipo: 'op'},
} as const;

// Aristas: [desde, hasta, etiqueta, t?, off?]
// t   = posición de la etiqueta a lo largo de la arista (0 = origen, 1 = destino)
// off = separación perpendicular de la etiqueta respecto a su línea
// Ambos se ajustan en las diagonales, donde convergen varias aristas sobre el
// nodo ×, para que ningún texto caiga sobre un nodo u otra flecha.
type Arista = [keyof typeof NODOS, keyof typeof NODOS, string, number?, number?];

const FORWARD: Arista[] = [
  ['x', 'mul', ''],
  ['w', 'mul', ''],
  ['mul', 'suma', 'wx = 6'],
  ['b', 'suma', ''],
  ['suma', 'resta', 'ŷ = 7'],
  ['resta', 'cuad', 'e = −3'],
];

// Aristas del backward: [desde, hasta, gradiente que viaja, t?]
const BACKWARD: Arista[] = [
  ['cuad', 'resta', '∂L/∂e = 2e = −6'],
  ['resta', 'suma', '∂L/∂ŷ = −6'],
  ['suma', 'b', '∂L/∂b = −6', 0.62, 40],
  ['suma', 'mul', '∂L/∂(wx) = −6', 0.30, 34],
  ['mul', 'w', '∂L/∂w = −6·x = −12', 0.78, 62],
];

/** Nodo circular del grafo, con color según su tipo. */
const Nodo: React.FC<{cx: number; cy: number; texto: string; tipo: string}> = ({
  cx, cy, texto, tipo,
}) => (
  <g>
    <circle
      cx={cx}
      cy={cy}
      r={62}
      fill={tipo === 'param' ? NARANJA : tipo === 'op' ? '#eef2f7' : '#dbeafe'}
      stroke={TINTA}
      strokeWidth={3}
    />
    <text
      x={cx}
      y={cy + 10}
      textAnchor="middle"
      fontSize={tipo === 'op' ? 44 : 30}
      fontFamily="monospace"
      fontWeight="bold"
      fill={TINTA}
    >
      {texto}
    </text>
  </g>
);

/** Flecha animada entre dos nodos con una etiqueta que aparece al llegar.
 *
 * `carril` desplaza la línea perpendicularmente a su dirección. Como el
 * backward recorre las mismas aristas en sentido inverso, su perpendicular
 * apunta al lado opuesto: dándole un carril propio, las flechas y etiquetas
 * de ambas fases nunca se solapan. */
const Flecha: React.FC<{
  de: {cx: number; cy: number};
  a: {cx: number; cy: number};
  progreso: number;      // 0 → 1: cuánto de la flecha se ha dibujado
  color: string;
  etiqueta: string;
  carril?: number;       // desplazamiento perpendicular de la línea
  offEtiqueta?: number;  // desplazamiento perpendicular de la etiqueta
  posEtiqueta?: number;  // posición de la etiqueta a lo largo de la arista (0..1)
}> = ({de, a, progreso, color, etiqueta, carril = 0, offEtiqueta = 34,
       posEtiqueta = 0.5}) => {
  if (progreso <= 0) return null;
  // Acortar la línea para que no tape los círculos de los nodos
  const dx = a.cx - de.cx, dy = a.cy - de.cy;
  const dist = Math.hypot(dx, dy);
  const ux = dx / dist, uy = dy / dist;
  const px = -uy, py = ux;                  // perpendicular unitaria
  const ox = px * carril, oy = py * carril; // offset del carril
  const x1 = de.cx + ux * 70 + ox, y1 = de.cy + uy * 70 + oy;
  const x2 = de.cx + ux * (dist - 76) * progreso + ux * 70 * (1 - progreso) + ox;
  const y2 = de.cy + uy * (dist - 76) * progreso + uy * 70 * (1 - progreso) + oy;
  return (
    <g>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={7} strokeLinecap="round" />
      {/* punta de flecha */}
      <polygon
        points={`${x2},${y2} ${x2 - 22 * ux + 11 * uy},${y2 - 22 * uy - 11 * ux} ${x2 - 22 * ux - 11 * uy},${y2 - 22 * uy + 11 * ux}`}
        fill={color}
      />
      {progreso >= 0.95 && etiqueta && (
        <text
          x={x1 + (x2 - x1) * posEtiqueta + px * offEtiqueta}
          y={y1 + (y2 - y1) * posEtiqueta + py * offEtiqueta + 9}
          textAnchor="middle"
          fontSize={24}
          fontFamily="monospace"
          fontWeight="bold"
          fill={color}
        >
          {etiqueta}
        </text>
      )}
    </g>
  );
};

export const ForwardBackward: React.FC = () => {
  const frame = useCurrentFrame();

  // Línea de tiempo: frames 0-160 = forward, 170-340 = backward
  const pasoForward = frame / 26;          // una arista cada ~26 frames
  const pasoBackward = (frame - 170) / 32; // backward algo más lento

  const fase = frame < 165 ? 'FORWARD PASS — calcular valores' :
    'BACKWARD PASS — propagar gradientes (regla de la cadena)';
  const colorFase = frame < 165 ? AZUL : ROJO;

  return (
    <AbsoluteFill style={{backgroundColor: '#f6f8fb', fontFamily: 'sans-serif'}}>
      <svg width="1920" height="1080">
        {/* Título y fase actual */}
        <text x={960} y={90} textAnchor="middle" fontSize={52} fontWeight="bold" fill={TINTA}>
          Grafo computacional: L = (wx + b − y)²
        </text>
        <text x={960} y={160} textAnchor="middle" fontSize={38} fontWeight="bold" fill={colorFase}>
          {fase}
        </text>

        {/* Aristas del forward (azul) */}
        {FORWARD.map(([de, a, etiqueta, t, off], i) => (
          <Flecha
            key={`f${i}`}
            de={NODOS[de]}
            a={NODOS[a]}
            progreso={interpolate(pasoForward - i, [0, 1], [0, 1], {
              extrapolateLeft: 'clamp', extrapolateRight: 'clamp',
            })}
            color={AZUL}
            etiqueta={etiqueta}
            posEtiqueta={t}
            offEtiqueta={off}
          />
        ))}

        {/* Aristas del backward (rojo), en sentido inverso */}
        {frame >= 170 && BACKWARD.map(([de, a, etiqueta, t, off], i) => (
          <Flecha
            key={`b${i}`}
            de={NODOS[de]}
            a={NODOS[a]}
            progreso={interpolate(pasoBackward - i, [0, 1], [0, 1], {
              extrapolateLeft: 'clamp', extrapolateRight: 'clamp',
            })}
            color={ROJO}
            etiqueta={etiqueta}
            carril={52}
            offEtiqueta={off ?? 34}
            posEtiqueta={t}
          />
        ))}

        {/* Nodos (siempre visibles, encima de las flechas) */}
        {Object.values(NODOS).map((n, i) => <Nodo key={i} {...n} />)}

        {/* Resultado de la loss al final del forward */}
        {frame > 150 && (
          <text x={1570} y={500} textAnchor="middle" fontSize={40}
            fontFamily="monospace" fontWeight="bold" fill={TINTA}>
            L = 9
          </text>
        )}

        {/* Conclusión: los gradientes listos para el update */}
        {frame > 330 && (
          <text x={960} y={900} textAnchor="middle" fontSize={36}
            fontFamily="monospace" fontWeight="bold" fill={ROJO}>
            θ ← θ − η·∇L :  w ← 3 − η·(−12) ,  b ← 1 − η·(−6)
          </text>
        )}

        {/* Leyenda */}
        <g fontSize={26} fontFamily="sans-serif">
          <circle cx={120} cy={980} r={16} fill={NARANJA} stroke={TINTA} strokeWidth={2} />
          <text x={150} y={989} fill={TINTA}>parámetro entrenable</text>
          <line x1={460} y1={980} x2={540} y2={980} stroke={AZUL} strokeWidth={7} />
          <text x={560} y={989} fill={TINTA}>forward (valores)</text>
          <line x1={830} y1={980} x2={910} y2={980} stroke={ROJO} strokeWidth={7} />
          <text x={930} y={989} fill={TINTA}>backward (gradientes)</text>
        </g>
      </svg>
    </AbsoluteFill>
  );
};
