import glob
import os

from flask import *
from werkzeug.utils import secure_filename

from timelapse_processor import TimelapseProcessor
from bing_search import BingSearch

# This web server follows the tutorial here:
# https://www.freecodecamp.org/news/how-to-build-a-web-application-using-flask-and-deploy-it-to-the-cloud-3551c985e492/

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads/custom/"  # TODO: this is a hard copy and paste from TP.UPLOAD_FILES (no *)w
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


def allowed(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route("/")
def home():
    """
    :return: Response with homepage render code.
    """
    return render_template("home.html")


@app.route("/error")
def error():
    """
    :return: Response with error page render code.
    """
    return render_template("error.html")


@app.route("/play")
def play():
    """
    :return: Response with a playable output video.
    """
    return render_template("play.html")

@app.route('/handle_data', methods=['POST'])
def handle_data():
    """
    This will firstly clear anything in the uploads/custom directory and download all of the clients timelapse files
    into there.
    Next, it will generate a timelapse from this data based on the given preferences.
    This video will then overwrite static/output/output.mp4

    :return: Response play.html
    """

    # Firstly validate the input files we are given and save to the server directory.

    request_files = request.files.getlist('files')
    request_preferences = request.form['preferences']

    # Clear everything before

    for file in glob.glob(TimelapseProcessor.UPLOAD_FILES):
        os.remove(file)

    # Upload files from client into the uploads/custom folder

    for file in request_files:
        filename = secure_filename(file.filename)

        if not allowed(filename):
            return redirect(url_for('error'))

        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # Generate the video and serve it to the client as a response

    processor = TimelapseProcessor(request_preferences)
    processor.process_files()

    return redirect(url_for('play'))


@app.route('/handle_search', methods=['POST'])
def handle_search():
    """
    Will go to Bing and collect a given number of image links for a particular term and return it here.
    Then, this will be downloaded using the TimelapseProcessor and will generate a hyperlapse.
    This will overwrite static/output/output.mp4.
    Finally, the user will be redirected to the play page where they will be able to play this .mp4.
    :return: Response play.html
    """

    request_search = request.form['bing-search']
    request_frames = request.form['frame-count']

    urls = BingSearch().search(request_search, int(request_frames))

    processor = TimelapseProcessor("uniform-sampling")
    processor.process_hyperlapse(urls)

    return play()


if __name__ == '__main__':
    app.run(debug=True)
