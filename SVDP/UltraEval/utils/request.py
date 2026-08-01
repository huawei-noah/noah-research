REQUEST_RETURN_LENGTHS = {
    "loglikelihood": 1,
    "generate": 1,
}


class Request:
    def __init__(self, request_type, instances, params, raw_example):
        if request_type not in REQUEST_RETURN_LENGTHS.keys():
            raise NotImplementedError(
                "The request type {} is not implemented!".format(request_type)
            )

        self.request_type = request_type
        self.instances = instances
        self.params = params
        self.raw_example = raw_example
