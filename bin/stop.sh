
SHHOME=$(cd `dirname $0`; pwd)
BASEHOME=$(cd $SHHOME/..; pwd)
COMPOSEFILE=$BASEHOME/.nlu-compose.yml
CONFIGFILE=$BASEHOME/conf/server_conf.ini

cd $BASEHOME

function Arr_Env(){
        echo "" > $BASEHOME/.env
        while read line;do
                if [[ ${line:0:1} == "[" ]];then
                        continue
                else
                        echo ${line// /} >> $BASEHOME/.env
                fi
        done < $CONFIGFILE
}

function Stop_Container(){
	chmod +x $SHHOME/docker-compose
	if [ $# -ne 0 ]; then
		$SHHOME/docker-compose -f $COMPOSEFILE stop -t 60 $@
		$SHHOME/docker-compose -f $COMPOSEFILE rm $@
	else
		$SHHOME/docker-compose -f $COMPOSEFILE down -t 60
		if [ $? -ne 0 ]; then
			echo "停止失败，请检查磁盘空间 或者 Docker状态"
		fi
	fi
}

function Remove_Bert_tmp(){
#	bert_file=`cat .env | grep host_model_dir`
#	(rm -rf ${bert_file#*=}/BertModel/tmp*) > /dev/null 2>&1
#	if [ $? -ne 0 ];then
#		echo "删除Bert tmp文件夹失败, 请手动删除"
#	fi
	cd $BASEHOME/libs/bert_serving/
	rm -rf tmp*
	cd $BASEHOME
}


function check_user(){
        docker images > /dev/null 2>&1
        if [ $? -ne 0 ];then
                echo "请使用sudo 或者 具有权限的用户执行"
        exit 1
        fi
}
(check_user; Arr_Env; Stop_Container; Remove_Bert_tmp)
