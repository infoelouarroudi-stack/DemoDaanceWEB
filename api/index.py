import os
import sys

# Chemin vers la racine du projet (.. depuis /api)
APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, APP_DIR)

# IMPORTANT :
# - Ton fichier racine doit s'appeler app.py
# - Il doit contenir: app = Flask(__name__)
from app import app
