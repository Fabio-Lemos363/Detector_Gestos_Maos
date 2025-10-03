
# --- Importar as bibliotecas --- #
import cv2
import mediapipe as mp


class DetectorMaos:
    """Classe responsável pela detecção das mãos."""
    def __init__(self, modo=False, max_maos=2, deteccao_confianca=0.5, rastreio_confianca=0.5,
                 cor_pontos=(0, 0, 255), cor_conexoes=(255, 255, 255)):

        # --- Inicializar os parâmetros --- #
        self.modo = modo
        self.max_maos = max_maos
        self.deteccao_confianca = deteccao_confianca
        self.rastreio_confianca = rastreio_confianca
        self.cor_pontos = cor_pontos
        self.cor_conexoes = cor_conexoes

        # --- Inicializar os módulos de detecção das mãos --- #
        self.maos_mp = mp.solutions.hands
        # Compatível com Python 3.10
        self.maos = self.maos_mp.Hands(
            static_image_mode=self.modo,
            max_num_hands=self.max_maos,
            min_detection_confidence=self.deteccao_confianca,
            min_tracking_confidence=self.rastreio_confianca
        )

        # --- Função para desenhar os pontos nas mãos --- #
        self.desenho_mp = mp.solutions.drawing_utils
        self.desenho_config_pontos = self.desenho_mp.DrawingSpec(color=self.cor_pontos, thickness=2, circle_radius=2)
        self.desenho_config_conexoes = self.desenho_mp.DrawingSpec(color=self.cor_conexoes, thickness=2, circle_radius=2)

        # --- Para detectar movimento (deslizar) --- #
        self.ultimo_x = None

    def encontrar_maos(self, imagem, desenho=True):
        """Detectar a(s) mão(s) na imagem"""
        imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        self.resultado = self.maos.process(imagem_rgb)

        if self.resultado.multi_hand_landmarks:
            for pontos in self.resultado.multi_hand_landmarks:
                if desenho:
                    self.desenho_mp.draw_landmarks(
                        imagem,
                        pontos,
                        self.maos_mp.HAND_CONNECTIONS,
                        self.desenho_config_pontos,
                        self.desenho_config_conexoes
                    )
        return imagem

    def encontrar_pontos(self, imagem, mao_num=0, desenho=True):
        """Obter coordenadas dos pontos da mão"""
        lista_pontos = []
        if self.resultado.multi_hand_landmarks:
            mao = self.resultado.multi_hand_landmarks[mao_num]
            altura, largura, _ = imagem.shape

            for id, ponto in enumerate(mao.landmark):
                centro_x, centro_y = int(ponto.x * largura), int(ponto.y * altura)
                lista_pontos.append([id, centro_x, centro_y])

                if desenho:
                    cv2.circle(imagem, (centro_x, centro_y), 3, (255, 0, 255), cv2.FILLED)
        return lista_pontos

    def reconhecer_gesto(self, lista_pontos):
        """Detecta gestos básicos: abrir, fechar, deslizar"""
        if len(lista_pontos) == 0:
            return None

        gesto = None

        # --- Abrir mão (dedos esticados) ---
        dedos = []
        # Polegar (ponta é 4, articulação é 3)
        if lista_pontos[4][1] > lista_pontos[3][1]:
            dedos.append(1)
        else:
            dedos.append(0)
        # Outros dedos (ponta maior que articulação de baixo)
        for tip in [8, 12, 16, 20]:
            if lista_pontos[tip][2] < lista_pontos[tip - 2][2]:
                dedos.append(1)
            else:
                dedos.append(0)

        total_dedos = dedos.count(1)

        if total_dedos == 5:
            gesto = "Mão Aberta"
        elif total_dedos == 0:
            gesto = "Punho Fechado"

        # --- Deslizar esquerda/direita ---
        x_punho = lista_pontos[0][1]  # landmark 0 = punho
        if self.ultimo_x is not None:
            diferenca = x_punho - self.ultimo_x
            if diferenca > 40:  # Threshold
                gesto = "Deslizar Direita"
            elif diferenca < -40:
                gesto = "Deslizar Esquerda"
        self.ultimo_x = x_punho

        return gesto


def main():
    cap = cv2.VideoCapture(0)
    detector = DetectorMaos()

    while True:
        _, imagem = cap.read()
        imagem = cv2.flip(imagem, 1)

        imagem = detector.encontrar_maos(imagem)
        lista_pontos = detector.encontrar_pontos(imagem, desenho=False)

        gesto = detector.reconhecer_gesto(lista_pontos)
        if gesto:
            cv2.putText(imagem, gesto, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Captura", imagem)
        if cv2.waitKey(1) & 0xFF == 27:  # tecla ESC para sair
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

