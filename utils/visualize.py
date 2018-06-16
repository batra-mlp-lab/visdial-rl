import os.path as pth
import json
import numpy as np
import visdom


class VisdomVisualize():
    def __init__(self,
                 env_name='main',
                 server="http://127.0.0.1",
                 port=8893,
                 enable=True):
        '''
            Initialize a visdom server on server:port
        '''
        print("Initializing visdom env [%s]" % env_name)
        self.is_enabled = enable
        if self.is_enabled:
            self.viz = visdom.Visdom(
                port=port,
                env=env_name,
                server=server,
            )
        else:
            self.viz = None
        self.wins = {}

    def linePlot(self, x, y, key, line_name, xlabel="Iterations"):
        '''
            Add or update a line plot on the visdom server self.viz
            Argumens:
                x : Scalar -> X-coordinate on plot
                y : Scalar -> Value at x
                key : Name of plot/graph
                line_name : Name of line within plot/graph
                xlabel : Label for x-axis (default: # Iterations)

            Plots and lines are created if they don't exist, otherwise
            they are updated.
        '''
        key = str(key)
        if self.is_enabled:
            if key in self.wins.keys():
                self.viz.line(
                    X = np.array([x]),
                    Y = np.array([y]),
                    win = self.wins[key],
                    update = 'append',
                    name = line_name,
                    opts = dict(showlegend=True),
                )
            else:
                self.wins[key] = self.viz.line(
                    X = np.array([x]),
                    Y = np.array([y]),
                    win = key,
                    name = line_name,
                    opts = {
                        'xlabel': xlabel,
                        'ylabel': key,
                        'title': key,
                        'showlegend': True,
                        # 'legend': [line_name],
                    }
                )

    def showText(self, text, key):
        '''
        Created a named text window or updates an existing one with
        the name == key
        '''
        key = str(key)
        if self.is_enabled:
            win = self.wins[key] if key in self.wins else None
            self.wins[key] = self.viz.text(text, win=win)

    def addText(self, text):
        '''
        Adds an unnamed text window without keeping track of win id
        '''
        if self.is_enabled:
            self.viz.text(text)
