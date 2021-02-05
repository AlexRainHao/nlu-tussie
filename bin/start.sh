SHHOME=$(cd `dirname $0`; pwd)
BASEHOME=$(cd $SHHOME/..; pwd)
COMPOSEFILE=$BASEHOME/.ner-compose.yml


cd $BASEHOME

function check_user(){
        docker images > /dev/null 2>&1
        if [ $? -ne 0 ];then
                echo "请使用sudo 或者 具有权限的用户启动"
        exit 1
        fi
}

function Arr_Env(){
	echo "" > $BASEHOME/.env
	while read line;do
        	if [[ ${line:0:1} == "[" ]];then
                	continue
	        else
        	        echo ${line// /} >> $BASEHOME/.env
	        fi
	done < $BASEHOME/conf/server_conf.ini
}

function Arr_Compose(){
	echo "version: '3.0'" > $COMPOSEFILE
	echo "services:" >> $COMPOSEFILE
	for ymlFile in `ls $BASEHOME/conf/compose`;do
		cat $BASEHOME/conf/compose/$ymlFile | grep -v "^version" | grep -v "^services" >> $COMPOSEFILE
	done
}

function Start_Image(){
	chmod +x $SHHOME/docker-compose
	cd $BASEHOME
	$SHHOME/docker-compose -f $BASEHOME/.ner-compose.yml up -d
}


(check_user; Arr_Env; Arr_Compose; Start_Image)




