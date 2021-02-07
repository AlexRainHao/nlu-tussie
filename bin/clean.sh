# server
server_id=`docker images | grep nlu-server | awk '{print $3}'`


for id in $modelploy_id $server_id;do
docker rmi $id
done
