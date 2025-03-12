class Processor:
    def __init__(self, Data_pass):
        self.name = name
        self.tasks = []

    def add_task(self, task):
        self.tasks.append(task)

    def process(self):
        print(f"Processor {self.name} is processing tasks")
        for task in self.tasks:
            task.execute()
        self.tasks = []
