__all__ = ["StartStreamLit", ]

from subprocess import Popen, PIPE
from tai_chi_engine.app.streamlit import run
import os
from io import StringIO
import sys


class StartStreamLit:
    def __init__(self, project: str, port: int = 8501):
        self.port = port
        self.process = None
        self.project = project
        self.run_script  = run.__file__
    
    def start(self):
        os.environ['TAI_CHI_ENGINE_APP'] = self.project
        self.process = Popen(
            ["streamlit", "run", self.run_script, "--server.port", str(self.port)],
            stdout=PIPE,
            stderr=PIPE,
        )

    def stop(self):
        self.process.terminate()
        self.process.wait()

    def follow(self):
        """
        Reads the output from the server process, and writes it to stdout.
        """
        for line in iter(self.process.stdout.readline, b''):
            sys.stdout.write(line.decode("utf-8"))