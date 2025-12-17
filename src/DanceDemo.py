import numpy as np
import cv2
import os
import time

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton

from GenNearest import GenNeirest
from GenVanillaNN import GenVanillaNN, SkeToImageTransform
from GenGAN import GenGAN


class DanceDemo:
    """
    Demo: affiche SOURCE VIDEO | SQUELETTE | GENERATION
    """
    def __init__(self, filename_src, typeOfGen=4):
        # cible: dataset du danseur "target" (sert aussi pour cropAndSke)
        self.target = VideoSkeleton("data/taichi1.mp4")
        # source: vidéo dont on prend les poses
        self.source = VideoReader(filename_src)

        self.typeOfGen = typeOfGen
        if typeOfGen == 1:
            print("Generator: GenNeirest")
            self.generator = GenNeirest(self.target)

        elif typeOfGen == 2:
            print("Generator: GenVanillaNN (Vecteur 26D -> Image)")
            self.generator = GenVanillaNN(self.target, loadFromFile=True, optSkeOrImage=1)

        elif typeOfGen == 3:
            print("Generator: GenVanillaNN (Stickman Image -> Image)")
            self.generator = GenVanillaNN(self.target, loadFromFile=True, optSkeOrImage=2)

        elif typeOfGen == 4:
            print("Generator: GenGAN")
            self.generator = GenGAN(self.target, loadFromFile=True)

        else:
            raise ValueError(f"DanceDemo: typeOfGen inconnu: {typeOfGen}")

        # affichage
        self.win_name = "Deep Dance Transfer"
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)

        # tailles panneaux
        self.H = 256
        self.W_panel = 256  # largeur d'un panneau
        self.W_total = self.W_panel * 3

        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def _ensure_uint8_bgr(self, img):
        """
        Standardise l'image pour cv2.imshow:
        - sort en uint8 BGR (H,W,3).
        - accepte float [0,1] ou float [0,255] ou uint8.
        """
        if img is None:
            return np.zeros((self.H, self.W_panel, 3), dtype=np.uint8)

        if img.dtype == np.uint8:
            out = img
        else:
            x = img
            # si float (probablement [0,1])
            if x.max() <= 1.5:
                x = x * 255.0
            out = np.clip(x, 0, 255).astype(np.uint8)

        # si grayscale par erreur
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

        return out

    def _make_panel(self, title, bgr_img):
        """
        Ajoute un titre en haut du panneau.
        """
        panel = cv2.resize(bgr_img, (self.W_panel, self.H))
        # bandeau titre
        cv2.putText(panel, title, (10, 25), self.font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return panel

    def draw(self):
        ske = Skeleton()    

        # image rouge en cas d'erreur squelette
        err_panel = np.zeros((self.H, self.W_panel, 3), dtype=np.uint8)
        err_panel[:] = (0, 0, 255)

        # pour FPS
        fps = 0.0
        t_last = time.time()

        print("Début de l'animation... 'q' quitter, 'n' sauter ~100 frames.")

        for i in range(self.source.getTotalFrames()):
            image_src = self.source.readFrame()
            if image_src is None:
                break

            # On ne met à jour squelette/génération qu'une frame sur 5
            # (tu peux changer 5 -> 1 si tu veux tout calculer)
            if i % 5 != 0:
                continue

            # crop + extraction skeleton (utilise mediapipe via VideoSkeleton)
            isSke, image_src_crop, ske = self.target.cropAndSke(image_src, ske)

            if not isSke:
                left = self._make_panel("SOURCE VIDEO", self._ensure_uint8_bgr(image_src_crop))
                mid = self._make_panel("SQUELETTE", err_panel)
                right = self._make_panel("GENERATION", err_panel)
                frame = np.hstack([left, mid, right])
                cv2.imshow(self.win_name, frame)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue

            # ---------------------------
            # 1) SOURCE VIDEO (gauche)
            # ---------------------------
            # image_src_crop est déjà BGR (cv2)
            left = self._make_panel("SOURCE VIDEO", self._ensure_uint8_bgr(image_src_crop))

            # ---------------------------
            # 2) SQUELETTE (milieu)
            # ---------------------------
            ske_img = np.zeros((self.H, self.W_panel, 3), dtype=np.uint8)
            # dessine le squelette (en BGR sur l'image)
            ske.draw(ske_img)
            mid = self._make_panel("SQUELETTE", ske_img)

            # ---------------------------
            # 3) GENERATION (droite)
            # ---------------------------
            out = self.generator.generate(ske)

            # Certains generate() renvoient uint8 BGR, d'autres float.
            out_bgr = self._ensure_uint8_bgr(out)
            right = self._make_panel("GENERATION", out_bgr)

            # concat final
            frame = np.hstack([left, mid, right])

            # FPS
            t_now = time.time()
            dt = t_now - t_last
            t_last = t_now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            cv2.putText(frame, f"FPS: {fps:.0f}", (10, 50), self.font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow(self.win_name, frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            if key & 0xFF == ord('n'):
                self.source.readNFrames(100)

        cv2.destroyAllWindows()


if __name__ == '__main__':
    # 1: nearest
    # 2: vanilla (26->img)
    # 3: vanilla (stick->img)
    # 4: GAN
    GEN_TYPE =2
    ddemo = DanceDemo("data/taichi2.mp4", GEN_TYPE)
    ddemo.draw()
