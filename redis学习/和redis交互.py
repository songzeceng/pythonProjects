from redis import StrictRedis

if __name__ == '__main__':
    redis = StrictRedis()

    redis.set(name="name", value="szc")
    print(redis.get("name"))
    redis.set(name="name", value="songzeceng")
    print(redis.get("name"))
    print(redis.delete("name"))

    keys = redis.keys()
    print(keys)

