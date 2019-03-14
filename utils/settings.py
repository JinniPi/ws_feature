"""
Setting connect database
"""
import os
from os.path import join, dirname
from dotenv import load_dotenv

load_dotenv()

DATA_MODEL_DIR = join(dirname(__file__), '../', os.environ.get("DATA_MODEL_DIR"))