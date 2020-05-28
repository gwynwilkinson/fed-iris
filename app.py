import io
from flask import Flask, request, render_template, Response, session
from flask_session import Session
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_svg import FigureCanvasSVG

from deeplearningmodel import *

#arg class defined in deeplearningmodel
args = Arguments()

# set up flask object and initialise new session to store data
app = Flask(__name__)
# Check Configuration section for more details
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)

# homepage route
@app.route("/")
def main():
	return render_template('index.html')

# form request handling route
@app.route("/echo", methods=['POST'])
def echo():
	
	# initialise f1 score list and set no. of workers
	session["f1_score"]=list()
	args.workers = int(request.form['workers'])
	
	# train & test 10 times and pass arguments
	for i in range(10):
		session["f1_score"].append(run_test(args))
	
	# return same page with new f1 scores inserted
	return render_template('index.html', score=session["f1_score"])

# graph image rendering route
@app.route("/matplot.svg")
def plot_graph():
	
	# load f1 scores from session into local list
	score_f1 = score=session["f1_score"]

	# create matplotlib graph showing f1 scores
	fig = Figure()
	axis = fig.add_subplot(1, 1, 1)
	x_points = range(len(score_f1))
	axis.plot(x_points, [score_f1[x] for x in x_points])
	
	# initialise raw output for rendering graph as svg data
	output = io.BytesIO()
	FigureCanvasSVG(fig).print_svg(output)
	
	# returns svg formatted graph
	return Response(output.getvalue(), mimetype="image/svg+xml")


# Flask requires this bit to run
if __name__ == "__main__":
    app.run()
