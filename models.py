# !/usr/bin/env python
# -*- coding:utf-8 -*-
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

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


class Car(db.Model):
    __tablename__ = 'car'
    id = db.Column(db.Integer, primary_key=True)
    license_plate = db.Column(db.String(40))
    type = db.Column(db.String(40))
    enter_time = db.Column(db.String(40))
    out_time = db.Column(db.String(40))
    cost = db.Column(db.Float)

    def __init__(self, license_plate, type, enter_time, out_time, cost):
        self.license_plate = license_plate
        self.type = type
        self.enter_time = enter_time
        self.out_time = out_time
        self.cost = cost

    def __repr__(self):
        return '<Car %r %r %r %r %r>'.format(self.license_plate,
                                             self.type,
                                             self.enter_time,
                                             self.out_time,
                                             self.cost)


# db.create_all()

if __name__ == '__main__':
    app.run()
