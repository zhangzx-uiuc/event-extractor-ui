from dataclasses import dataclass
from flask import Flask, render_template, jsonify, request, make_response, send_from_directory
from flask_bootstrap import Bootstrap
from flask_moment import Moment
from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField
from wtforms.validators import DataRequired
from flask_wtf.file import FileField, FileAllowed, FileRequired
from werkzeug.utils import secure_filename
from flask import redirect, url_for
from flask_cors import CORS
from model import ZeroShotModel
# from tornado.wsgi import WSGIContainer
# from tornado.httpserver import HTTPServer
# from tornado.ioloop import IOLoop
from api import load_ckpt, annotate, annotate_arguments
from generate_weakly_supervised_data import generate
from run_train import quick_train
from utils.options import parse_arguments
from transformers import BertTokenizerFast

from datetime import datetime
import os
import sys
import glob
import json
import torch
import spacy
from nltk.tokenize import TreebankWordTokenizer

# Hyper Params
torch.cuda.manual_seed(100)
torch.manual_seed(100)
torch.cuda.manual_seed_all(100)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
bootstrap = Bootstrap(app)
moment = Moment(app)
cors = CORS(app)

@dataclass
class Options(object):
    gpu:int
    seed:int
    batch_size:int
    eval_batch_size:int

class DataHandler(object):
    def __init__(self, root_dir:str):
        self.root_dir = root_dir
        self.ed = None

    def run_weak_supervision(self,force=False):
        if not os.path.exists(os.path.join(self.root_dir, 'data')):
            os.makedirs(os.path.join(self.root_dir, 'data'))
        generate(
            train_file=os.path.join(self.root_dir, "train.json"),
            dev_file=os.path.join(self.root_dir, "dev.json"),
            label_json=os.path.join(self.root_dir, "label_info.json"),
            example_json=os.path.join(self.root_dir, "examples.json"),
            model_name='bert-large-cased',
            output_save_dir=os.path.join(self.root_dir, 'data'),
            corpus_jsonl=os.path.join(self.root_dir, "corpus.json"),
            force=force
        )

        with open(os.path.join(self.root_dir, "label_info.json"), 'r', encoding='utf-8') as f:
            label_info = json.loads(f.read())
        ontology = {}
        for type in label_info:
            ontology[type] = label_info[type]["roles"]
        self.ontology = ontology
        self.generate_ontology()

    def run_train(self):
        opts = parse_arguments()
        opts.root = self.root_dir
        opts.json_root = self.root_dir
        opts.example_regularization=True
        opts.log_dir = os.path.join(self.root_dir, 'log')
        opts.log = os.path.join(opts.log_dir, f"logfile.log")
        if not os.path.exists(opts.log_dir):
            os.makedirs(opts.log_dir)
        opts.train_epoch = 2
        quick_train(opts)
        self.load_argument_model()
        return os.path.join(self.root_dir, 'log', 'model.best')
    
    def generate_ontology(self):
        event_type_idxs, role_type_idxs = {}, {"unrelated object": -1}
        event_num, role_num = 0, 0
        for event_type in self.ontology:
            if event_type not in event_type_idxs:
                event_type_idxs[event_type] = event_num
                event_num += 1

        for event_type in self.ontology:
            roles = self.ontology[event_type]
            for role in roles:
                if role not in role_type_idxs:
                    role_type_idxs[role] = role_num
                    role_num += 1
        self.event_type_idxs = event_type_idxs
        self.role_type_idxs = role_type_idxs

    def load_model(self, ed_path):
        self.ed = load_ckpt(ed_path)
    
    def load_argument_model(self):
        arg_path = os.path.join(self.root_dir, "checkpoint.pt")
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-large-uncased")

        arg_model = ZeroShotModel("bert-large-uncased", self.role_type_idxs, self.role_type_idxs, self.ontology, self.ontology, self.event_type_idxs, self.event_type_idxs, 0.3, 1024, 256, 128, 128, 0.1, "cpu")
        arg_model.to("cpu")
        arg_model.compute_train_role_reprs(self.tokenizer)
        arg_model.compute_test_role_reprs(self.tokenizer)

        arg_model.load_state_dict(torch.load(arg_path))
        arg_model.eval()
        self.arg_model = arg_model
        self.spacy_model = spacy.load("en_core_web_sm")
        self.nltk_tokenizer = TreebankWordTokenizer()

    def visualize(self, sentence, annotation):
        trigger = annotation['trigger']
        arguments = annotation['arguments']
        spans = arguments + [[trigger[0], trigger[1], 'Trigger']]
        spans.sort(key=lambda t:t[0])

        sentence_pieces = []
        last = 0
        for span_start, span_end, span_label in spans:
            if span_start > last:
                sentence_pieces.append([sentence[last:span_start], None])
            sentence_pieces.append([sentence[span_start:span_end], span_label])
            last = span_end
        if span_end < len(sentence):
            sentence_pieces.append([sentence[span_end:], None])

        vis = ''
        for sent_piece in sentence_pieces:
            if sent_piece[1] is None:
                vis += f"<span>{sent_piece[0]}</span>"
            elif sent_piece[1] == 'Trigger':
                vis += f'[<span style="background-color:violet">{sent_piece[0]}</span>]<sub>Trigger</sub>'
            else:
                vis += f'[<span style="background-color:#22ffff">{sent_piece[0]}</span>]<sub>{sent_piece[1]}</sub>'
        return f'<p><b>{trigger[2]}</b> event triggered by "<span style="background-color:violet">{sentence[trigger[0]:trigger[1]]}</span>"</p>\n<p>{vis}</p>'
    
dh = DataHandler(root_dir=os.path.join(app.instance_path, 'data'))

class FileForm(FlaskForm):
    label_file = FileField('Select a label file', validators=[FileRequired()])
    example_file = FileField('Select an example file', validators=[FileRequired()])
    train_file = FileField('Select a train corpus file', validators=[FileRequired()])
    dev_file = FileField('Select a dev corpus file', validators=[])
    submit = SubmitField('Start training')

class DetectForm(FlaskForm):
    sentence = StringField('Input a sentence', validators=[DataRequired()])
    submit = SubmitField('Submit')

class ModelForm(FlaskForm):
    model_file = FileField('Select a model file to upload', validators=[FileRequired()])
    submit = SubmitField('Load Model')

@app.route('/', methods=['GET', 'POST'])
def index():
    global dh
    form = FileForm()
    if not os.path.exists(os.path.join(app.instance_path, 'data')):
        os.makedirs(os.path.join(app.instance_path, 'data'))
    if form.validate_on_submit():
        lf = form.label_file.data
        lname = secure_filename(lf.filename)
        lf.save(os.path.join(
            app.instance_path, 'data', 'label_info.json'
        ))

        ef = form.example_file.data
        ename = secure_filename(ef.filename)
        ef.save(os.path.join(
            app.instance_path, 'data', 'examples.json'
        ))

        tf = form.train_file.data
        tname = secure_filename(tf.filename)
        tf.save(os.path.join(
            app.instance_path, 'data', 'train.json'
        ))
        if form.dev_file.data is None:
            with open(os.path.join(app.instance_path, 'data', 'train.json')) as fp:
                train = json.load(fp)
                train = train[:int(len(train)*0.89)]
                dev = train[int(len(train)*0.89):]
            with open(os.path.join(app.instance_path, 'data', 'train.json'), 'wt') as fp:
                json.dump(train, fp, indent=4)
            with open(os.path.join(app.instance_path, 'data', 'dev.json'), 'wt') as fp:
                json.dump(dev, fp, indent=4)
        else:
            df = form.dev_file.data
            dname = secure_filename(df.filename)
            df.save(os.path.join(
                app.instance_path, 'data', 'dev.json'
            ))
        dh.run_weak_supervision()
        mp = dh.run_train()
        dh.load_model(mp)

        # with open(os.path.join(dh.root_dir, "label_info.json"), 'r', encoding='utf-8') as f:
        #     label_info = json.loads(f.read())
        # ontology = {}
        # for type in label_info:
        #     ontology[type] = label_info[type]["roles"]
        # dh.ontology = ontology
        # dh.generate_ontology()
        # dh.load_argument_model()

        return redirect(url_for('detect'))
    return render_template('index.html', form=form)

@app.route("/detect", methods=['GET', 'POST'])
def detect():
    global dh    

    form = DetectForm()
    model_form = ModelForm()
    if form.validate_on_submit():
        sent = form.sentence.data

        trigger_annotations = annotate([sent], dh.ed)
        annotations = annotate_arguments(trigger_annotations, sent, dh.arg_model, dh.tokenizer, dh.spacy_model, dh.nltk_tokenizer)
        # annotations = [[{"trigger": [80, 86, "Transaction:Transfer-Money"], "arguments": [[20, 30, "Giver"], [87, 92, "Recipient"]]}]]
        print(annotations)
        visualized = []
        for annotation in annotations[0]:
            visualized.append(dh.visualize(sent, annotation))
        return render_template('detect.html', form=form, model_form=model_form, results=visualized)
    if model_form.validate_on_submit():
        mf = model_form.model_file.data
        mpath = os.path.join(
            app.instance_path, 'data', 'log', 'model.best'
        )
        mf.save(mpath)
        dh.load_model(mpath)
        return render_template('detect.html', form=form, model_form=model_form, results=None)
    return render_template('detect.html', form=form, model_form=model_form, results=None)


@app.route('/download', methods=['GET', 'POST'])
def download():
    uploads = os.path.join(app.instance_path, 'data', 'log')
    return send_from_directory(directory=uploads, path='model.best')


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html')

@app.errorhandler(500)
def internel_server_error(e):
    return render_template('500.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)