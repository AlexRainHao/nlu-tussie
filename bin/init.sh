SHHOME=$(cd `dirname $0`; pwd)
BASEHOME=$(cd $SHHOME/..; pwd)


function load(){
	for app in `ls $BASEHOME/images`;do
		echo "load $app"
		docker load --input $BASEHOME/images/$app
	done
}


function check_user(){
	docker images > /dev/null 2>&1
	if [ $? -ne 0 ];then
		echo "请使用sudo 或者 具有权限的用户启动"
	exit 1
	fi
}


(check_user; load)
