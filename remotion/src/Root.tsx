// Raíz de composiciones: declara los videos disponibles, su duración,
// resolución y fps. `npm run studio` las lista en el navegador.
import {Composition} from 'remotion';
import {ForwardBackward} from './ForwardBackward';
import {FlujoAtencion} from './FlujoAtencion';

export const RemotionRoot: React.FC = () => {
  return (
    <>
      {/* Sesión 1: forward pass y backpropagation sobre el grafo
          computacional de L = (wx + b − y)² */}
      <Composition
        id="ForwardBackward"
        component={ForwardBackward}
        durationInFrames={360}   // 12 s a 30 fps
        fps={30}
        width={1920}
        height={1080}
      />
      {/* Sesión 3: pipeline de scaled dot-product attention */}
      <Composition
        id="FlujoAtencion"
        component={FlujoAtencion}
        durationInFrames={420}   // 14 s a 30 fps
        fps={30}
        width={1920}
        height={1080}
      />
    </>
  );
};
