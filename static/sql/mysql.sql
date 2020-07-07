create database lpr;

CREATE TABLE car
(
    id            INTEGER NOT NULL AUTO_INCREMENT,
    license_plate VARCHAR(40),
    type         VARCHAR(40),
    enter_time    VARCHAR(40),
    out_time      VARCHAR(40),
    cost          FLOAT,
    PRIMARY KEY (id)
);