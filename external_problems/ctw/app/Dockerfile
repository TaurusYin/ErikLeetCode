FROM centos:7.9.2009
RUN yum install -y wget
RUN wget -P /etc/yum.repos.d/ http://mirrors.aliyun.com/repo/Centos-7.repo && yum install -y epel-release && yum install -y python3 python3-devel gcc gcc-c++ mysql mysql-devel MySQL-python python3-tkinter readline-devel libcurl libcurl-devel
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
# RUN pip3 install --user ray==2.1.0 && pip3 install --user torch  && pip3 install "ray[serve]" transformers requests sentencepiece accelerate
RUN mkdir -p /code/
WORKDIR /code/ml-assignment/app/service/
COPY ./ml-assignment/ /code/ml-assignment/
RUN pip3 install --upgrade pip && pip3 install -r /code/ml-assignment/app/requirements.txt
ENV PYTHONPATH "${PYTHONPATH}:/code/ml-assignment/"
ENTRYPOINT ["sh", "/code/ml-assignment/app/entrypoint.sh"]