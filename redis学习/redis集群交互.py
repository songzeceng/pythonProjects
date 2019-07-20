from rediscluster import StrictRedisCluster

if __name__ == '__main__':
    redis = StrictRedisCluster(startup_nodes=[
        {"host": "", "port": ""},
        {"host": "", "port": ""},
        {"host": "", "port": ""}
    ], decode_responses=True)
    redis.set(name="name", value="szc")
    print(redis.get("name"))