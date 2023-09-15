import sys

def error_message(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = f"Error occured in python script name {file_name}, line number: {exc_tb.tb_lineno}, error message: {error}."

    return error_message

class CustomException(Exception):

    def __init__(self, error_message, error_details:sys):
        super().__init__(error_message)

        self.error_message = error_message(error_message, error_details)
    def __str__(self) -> str:
        return self.error_message