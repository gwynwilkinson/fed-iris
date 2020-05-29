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
	
	# Get no. of workers from form
	args.workers = int(request.form['workers'])
	
	# Run federated model training to get F1 scores list
	session["f1_scores"] = run_test(args)
	
	# return same page with new F1 scores inserted
	return render_template('index.html', score=session["f1_scores"])

# graph image rendering route
@app.route("/matplot.svg")
def plot_graph():
	
	# load f1 scores from session into local list
	score_f1 = score=session["f1_scores"]

	# create matplotlib graph showing f1 scores
	fig = Figure()
	axis = fig.add_subplot(1, 1, 1)
	x_points = range(len(score_f1))
	axis.plot(x_points, [score_f1[x] for x in x_points])
	axis.set_title('Federated Model - F1 Score History')
	axis.set_ylabel('F1 Score')
	axis.set_xlabel('Epoch')
	
	# initialise raw output for rendering graph as svg data
	output = io.BytesIO()
	FigureCanvasSVG(fig).print_svg(output)
	
	# returns svg formatted graph
	return Response(output.getvalue(), mimetype="image/svg+xml")


# Flask requires this bit to run
if __name__ == "__main__":
    app.run()
