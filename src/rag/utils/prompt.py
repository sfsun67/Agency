from pydantic import BaseModel

class BasePrompt(BaseModel):
    prompt_template: str

    def render_prompt(self) -> str:
        """method for rendering the variables (self.var_name) in prompt_template"""
        return self.prompt_template.format(self=self)
    
class ExamplePrompt(BasePrompt):
    var_name_1: str
    var_name_2: str
    prompt_template: str = "this is an example prompt with slot {self.var_name_1} and {self.var_name_2}"
    
    
if __name__ == "__main__":
    prompt = ExamplePrompt(
        var_name_1="var1",
        var_name_2="var2",
    )
    print(prompt.render_prompt())