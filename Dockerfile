#FROM centos:7.6.1810
FROM nerenv:latest
MAINTAINER ASR_BASE
LABEL description="Yunjia_NER"
LABEL maintainer="Yuan Yuhao <yuanyh@yunjiacloud.com>"

## timezone
#RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
#    echo 'Asia/Shanghai' >/etc/timezone
#
## install dependency
#WORKDIR /home/admin
#COPY build_deped/*.repo ./
##RUN mkdir -p /etc/yum.repos.bak && mv /etc/yum.repos.d/*.repo /etc/yum.repos.bak && \
##    mv *.repo /etc/yum.repos.d/ && \
##    yum -y clean all && yum makecache && yum install -y \
##    python36-setuptools python36-pip \
##    gcc \
##    kde-l10n-Chinese glibc-common && \
##    yum -y clean all
#
#RUN yum install -y python36-setuptools python36-pip gcc kde-l10n-Chinese glibc-common
#RUN yum -y clean all
#
#RUN localedef -c -f UTF-8 -i zh_CN zh_CN.utf8 && \
#    export PYTHONIOENCODING=utf-8

# python requirement
RUN yum install -y libXext libSM libXrender
COPY build_deped/requirement.txt requirement.txt

RUN pip3 install -U pip setuptools -i https://mirror.baidu.com/pypi/simple --no-cache-dir && \
    pip3 install -r requirement.txt -i https://mirror.baidu.com/pypi/simple --no-cache-dir

# server requirement
WORKDIR /home/admin/NER_Server

COPY start_model.sh start_model.sh

RUN ["chmod", "+x", "/home/admin/NER_Server/start_model.sh"]

ENTRYPOINT ["/bin/bash", "-c", "sh /home/admin/NER_Server/start_model.sh"]
