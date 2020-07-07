# !/usr/bin/env python
# -*- coding:utf-8 -*-
from flask import Flask, render_template, request, Response
from start import starting
from flask_sqlalchemy import SQLAlchemy
import models
from datetime import datetime, timedelta
import random
from werkzeug.utils import secure_filename
import tensorflow as tf
import shutil
import os
import urllib.request

# 保证能多次识别图片
global graph, model
graph = tf.get_default_graph()

app = Flask(__name__)
# 设置连接数据库的URL
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:282127@127.0.0.1:3306/lpr'
# 设置每次请求结束后会自动提交数据库中的改动
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
# 数据库和模型类同步修改
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
# 查询时会显示原始SQL语句
app.config['SQLALCHEMY_ECHO'] = False
db = SQLAlchemy(app)

# 上传图片并保存到本地
UPLOAD_FOLDER = 'static/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'gif', 'GIF'])
basedir = os.path.abspath(os.path.dirname(__file__))
file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    del_dir()
    if request.method == 'POST':
        f = request.files['file']
        c = request.form.get('came')

        if f and allowed_file(f.filename):
            fname = secure_filename(f.filename)
            ext = fname.rsplit('.', 1)[1]
            new_filename = 'upload' + '.' + ext
            f.save(os.path.join(file_dir, new_filename))
        elif c:
            return render_template('camera.html')
        else:
            return render_template('result_fail.html')

        type, pstr = get_pstr()

        if not pstr is None:
            update_date(type, pstr)
            filename = 'upload' + '.' + 'jpg'
            filename1 = 'upload' + str(random.randint(10, 1000)) + '.' + 'jpg'
            os.rename(file_dir + '/' + filename, file_dir + '/' + filename1)
            file_dir1 = os.path.join("/static/upload/", filename1)

            car = inquire_date(pstr)
            return render_template('result_success.html',
                                   car=car,
                                   url=file_dir1
                                   )
        return render_template('result_fail.html')


@app.route('/camera1', methods=['POST', 'GET'])
def camera1():
    del_dir()
    if request.method == 'POST':
        i = request.files['image']
        print(i)
        if i:
            f = '{}{}{}{}'.format(UPLOAD_FOLDER, os.sep, "upload", ".jpg")
            i.save('%s' % f)
            return Response("%s saved" % f)


@app.route('/camera2', methods=['POST', 'GET'])
def camera2():
    if request.method == 'POST':
        c = request.form.get('ca')
    type, pstr = get_pstr()

    if not pstr is None and not c is None:
        update_date(type, pstr)
        filename = 'upload' + '.' + 'jpg'
        filename1 = 'upload' + str(random.randint(10, 1000)) + '.' + 'jpg'
        os.rename(file_dir + '/' + filename, file_dir + '/' + filename1)
        file_dir1 = os.path.join("/static/upload/", filename1)

        car = inquire_date(pstr)
        return render_template('result_success.html',
                               car=car,
                               url=file_dir1
                               )
    return render_template('result_fail.html')


@app.route('/status', methods=['GET', 'POST'])
def status():
    car = models.Car.query.all()
    db.session.commit()
    return render_template('status.html', tem=car)


@app.route('/introduction', methods=['GET', 'POST'])
def introduction():
    return render_template('introduction.html')


def del_dir():
    if os.path.exists(file_dir):
        shutil.rmtree(file_dir)  # 将整个文件夹删除
        os.makedirs(file_dir)


def get_pstr():
    # 覆盖默认的图，防止每次启动程序只能识别一张图片
    with graph.as_default():
        type, pstr = starting("static/upload/upload.jpg")
    return type, pstr


# 更新数据库数据
def update_date(type, license_plate):
    from models import db

    tmp_time = random.randint(10, 100)
    enter_time = datetime.now()
    out_time = enter_time + timedelta(minutes=tmp_time)
    enter_time = enter_time.strftime("%Y-%m-%d %H:%M:%S")
    out_time = out_time.strftime("%Y-%m-%d %H:%M:%S")
    if tmp_time <= 60:
        cost = tmp_time * 0.05
    else:
        cost = 60 * 0.05 + (tmp_time - 60) * 0.1

    car = models.Car(license_plate, type, enter_time, out_time, cost)

    db.session.add(car)
    db.session.commit()


# 查询数据
def inquire_date(pstr):
    car = models.Car.query.filter_by(license_plate=pstr).first()
    return car


# 删除临时表数据　
def del_data():
    db.session.query(models.Car).filter(models.Car.id != -1).delete()
    db.session.commit()


if __name__ == '__main__':
    del_data()
    app.run(debug=True)
