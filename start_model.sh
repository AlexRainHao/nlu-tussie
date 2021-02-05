BIN_PATH=/home/admin/NER_Server
#BIN_PATH=/home/user/yuanyh/nerYJCloud
BERT_MODEL_PATH=$BIN_PATH/model/BertModel
ALBERT_MODEL_PATH=$BIN_PATH/model/AlbertModel
NER_MODEL_PATH=$BIN_PATH/model/NerModel

SRC_PATH=$BIN_PATH/libs

CONF_PATH=$BIN_PATH/conf
LOG_PATH=$BIN_PATH/logs


# ========================

function Init_Logger(){
        rm -rf $LOG_PATH/server.log
}

function Logger(){
        echo "[`date +%Y-%m-%d\ %H:%M`] $1" >> $LOG_PATH/server.log
}

function error_exit(){
        echo "[`date +%Y-%m-%d\ %H:%M`] $1" >> $LOG_PATH/server.log
        exit 1
}

# 1. bert server
function Stop_Bert(){
        for pid in `ps -ef | grep start_bert_server | grep -v grep | awk '{print $2}'`;do
                kill -9 $pid
        done
}



function Start_Albert(){
  cd $SRC_PATH/bert_serving
  Logger "Start Albert Server..."
  Stop_Bert

  # 0.1 start albert server
  nohup python3 start_bert_server.py -model_dir $ALBERT_MODEL_PATH > /dev/null 2>&1 &

  # 0.2 check status
  cd $BIN_PATH
  _start_time=$(date +%s)
  until `curl -X POST http://0.0.0.0:5557/encode \
  -H 'content-type: application/json' \
  -d '{"id":123,"texts":["您好"],"is_tokenized":false}' > /dev/null 2>&1`;do
    Logger "Checking Bert Server Status..."
    sleep 3s
    _end_time=$(date +%s)
    if [[ $((_end_time-_start_time)) -gt 120 ]];then
      Stop_Bert
                        error_exit "Start Albert Server Failed!!"
        fi
  done
        Logger "Start Albert Server Successfully"
        cd $BIN_PATH
}


function Start_Bert(){
  cd $BERT_MODEL_PATH
  Logger "Start Bert Server..."
  Stop_Bert

  # 1.1 start server
  nohup bert-serving-start \
  -model_dir $BERT_MODEL_PATH \
  -http_port 5557 \
  -max_seq_len 40 \
  -max_batch_size 256 \
  -http_max_connect 20 \
  -num_worker 4 >> $LOG_PATH/bert-serving.log &

  # 1.2 check status
  cd $BIN_PATH
  _start_time=$(date +%s)
  until `curl -X POST http://0.0.0.0:5557/encode \
  -H 'content-type: application/json' \
  -d '{"id":123,"texts":["您好"],"is_tokenized":false}' > /dev/null 2>&1`;do
    Logger "Checking Bert Server Status..."
    sleep 3s
    _end_time=$(date +%s)
    if [[ $((_end_time-_start_time)) -gt 120 ]];then
      Stop_Bert
                        error_exit "Start Bert Server Failed!!"
        fi
  done
        Logger "Start Bert Server Successfully"
        cd $BIN_PATH
}

# 2. NER server
function Start_Ner(){
  cd $SRC_PATH
  Logger "Start NER Server..."

  # 2.1 start server
  nohup python3 -m ner_yjcloud.server \
  -P 9001 \
  --path $NER_MODEL_PATH \
  --pre_load default \
  --response_log $LOG_PATH \
  > /dev/null 2>&1 &

  # 2.2 check status
  _start_time=$(date +%s)
  until `curl -X POST http://0.0.0.0:9001/parse \
  -d '{"q":"今天天气怎么样"}' > /dev/null 2>&1`; do
    Logger "Checking NER Status..."
    sleep 3s
    _end_time=$(date +%s)
    if [[ $((_end_time-_start_time)) -gt 120 ]];then
      for pid in ` ps -ef | grep ner_yjcloud.server | grep -v grep | awk '{print $2}'`;do
        kill -9 $pid
      done
      error_exit "Start NER Server Failed!! "
    fi
  done
  Logger "Start NER Server Successfully"
  cd $BIN_PATH
}

function HangUp_Server(){
        while true;do
                Logger "Waiting For Calling..."
        sleep 5m
        done
}

#(Init_Logger; Start_Bert; Start_Ner; HangUp_Server)
(Init_Logger; Start_Albert; Start_Ner; HangUp_Server)
#(Init_Logger; Start_Albert; Start_Ner)
