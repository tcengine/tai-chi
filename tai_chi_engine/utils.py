from tai_chi_tuna.utils import clean_name
from pathlib import Path
import pandas as pd

from tai_chi_tuna.config import PhaseConfig
from tai_chi_tuna.front.html import DOM, list_group_kv
from pathlib import Path

def df_creator_image_folder(path: Path) -> pd.DataFrame:
    """
    Create a dataframe ,
    Which list all the image path under a system folder
    """
    path = Path(path)
    files = []
    formats = ["jpg", "jpeg", "png"]
    for fmt in formats:
        files.extend(path.rglob(f"*.{fmt.lower()}"))
        files.extend(path.rglob(f"*.{fmt.upper()}"))
    return pd.DataFrame({"path": files}).sample(frac=1.).reset_index(drop=True)

def join_col_list(Columns):
    return ",".join(map(lambda x:f"'{x['src']}'", Columns))

class Narrator:
    """
    Under jupyter notebook: 
    ```python
    narrator = Narrator(phase:PhaseConfig)
    narrator()
    ```
    This will narrate how to train the model, in full detail
    """
    
    x_color = "#3399FF"
    y_color = "#33CC55"
    next_color = "#11CCDD"

    def __init__(self, phase: PhaseConfig):
        self.phase = phase

    def __repr__(self):
        return str(self)

    def __str__(self) -> str:
        return f"<Narrator>, run self() to print out in juypter"

    def __call__(self,):
        self.narrate()()

    def title(self, text) -> DOM:
        return DOM(text, "h3",
                   {"style": "color:#FFAA33;background-color:#FFFFFF;padding:3px"})

    def empty_step(self,) -> DOM:
        return DOM("", "ul", dict())

    def line(self, inner) -> DOM:
        return DOM(inner, "li", dict(style="padding:2px;background-color:#FFFFDD"))
    
    def ok(self, text) -> DOM:
        return DOM(text, "h5",
                   dict(style=f"background-color:{self.next_color};color:#FFF;padding:3px"))

    def narrate(self) -> DOM:
        """
        Chaining up 4 steps
        """
        doc = DOM("", "div")
        step1 = self.step_enrich()
        doc.append(step1)

        step2 = self.step_quantify()
        doc.append(step2)

        step3 = self.step_model()
        doc.append(step3)
        
        step4 = self.step_train()
        doc.append(step4)

        return doc

    def step_enrich(self) -> DOM:
        config = self.phase.config
        step = self.empty_step()
        step.append(self.title("Step 1: Enrich"))
        if "enrich" in config:
            enrichs = config['enrich']
            for enrich in enrichs:
                src = enrich.get('src')
                enrich_type = enrich.get('enrich')
                step.append(self.line(f"select column '{src}' for 'src'"))
                step.append(
                    self.line(
                        f"select '{enrich_type}' for 'enrich'"))
                step.append(
                    self.line(
                        f"The kwargs of '{enrich_type}' should be set to:"))
                step.append(self.line(list_group_kv(enrich.get("kwargs"))))
            step.append(
                self.ok("Congrats! now Click ✔️Next above"))
        else:
            step.append(
                self.ok("No enrich set, click ✔️Next above"))
        return step

    def step_quantify(self) -> DOM:
        config = self.phase.config
        step = self.empty_step()
        step.append(self.title("Step 2: Quantify"))
        if "quantify" in config:
            quantifies = config['quantify']
            Xs = list(filter(lambda x: x['x'], quantifies))
            Ys = list(filter(lambda x: x['x'] == False, quantifies))

            step.append(f"""
                In this model we are using 
                <strong style='color:{self.x_color}'>
                column(s):{join_col_list(Xs)}</strong> 
                to guess <strong style='color:{self.y_color}'>
                column:{join_col_list(Ys)}</strong>
                """)

            for is_x, Columns in zip([True, False], [Xs, Ys]):
                if is_x:
                    step.append(
                            f"""
                            <h4 style='color:{self.x_color}'>
                            There are {len(Xs)} X column(s):</h4>"""
                        )
                else:
                    step.append(
                            f"""
                            <h4 style='color:{self.y_color}'>
                            There is our {len(Ys)} Y column:</h4>"""
                        )
                for col in Columns:
                    step.append(
                        self.line(f"""
                        <h5>Quantify config for '{col['src']}' column</h5>""")
                    )
                    step.append(
                        self.line(f"""
                        Select '{col['src']}' as 'src'"""))
                    step.append(
                        self.line(f"""
                        Choose 'As {'X' if is_x else 'Y'}', 
                        So column {col['src']} is used as 
                        {'input' if is_x else 'output'} data"""))
                    step.append(
                        self.line(f"Click Button: 'Run Interact'"))
                    step.append(
                        self.line(f"""
                        Choose '{col['quantify']}' as the method to 
                        transform '{col['src']}' into matrix/tensor"""))
                    if 'kwargs' in col:
                        step.append(
                            self.line(
                                f"Click Button: 'Run Interact' to set kwargs"))
                        step.append(
                            self.line(
                                f"The kwargs of '{col['quantify']}' should be set to:"))
                        step.append(
                            self.line(list_group_kv(col["kwargs"])))
                        step.append(
                            self.line(f'Click Button:+Create'))
                    else:
                        step.append(
                            self.line(f"""
                            Click Button: 'Run Interact' then click '+Create'"""))
            step.append(
                self.ok("Congrats! now Click ✔️Next above")
            )
        return step

    def step_model(self) -> DOM:
        config = self.phase.config
        step = self.empty_step()
        step.append(self.title("Step 3: Model"))
        step.append("<h5>Batching up Data</h5>")

        batch_level = config['batch_level']
        valid_ratio = batch_level['valid_ratio']
        batch_size = batch_level['batch_size']
        shuffle = 'checked' if batch_level['shuffle'] else 'unchecked'
        num_workers = batch_level['num_workers']
        step.append(
            self.line(
                f"""We slide valid_ratio to '{valid_ratio}'
                    as around '{int(valid_ratio*100)}%' of total data
                    will be used as validation set""")
        )
        step.append(
            self.line(f"""
            We set batch_size to '{batch_size}',
            if you run into memory error or GPU OOM,
            keep halfing this number, until it works""")
        )
        step.append(
            self.line(f"We keep the 'shuffle' checkbox to '{shuffle}'")
        )
        step.append(
            self.line(
                f"""We set the 'num_workers' to '{num_workers}',
                so {1 if num_workers==0 else num_workers}
                cpu core(s) will transform the data.
                """,
            ))
        step.append(
            self.line("Click '+create'")
        )
        x_models = config["x_models"]
        y_models = config["y_models"]
        
        for xy, parts, color in zip(
            ["x","y"],
            [x_models, y_models],
            [self.x_color, self.y_color]):
            # going through model parts
            step.append(f"""
                <h4 id='step-model-{xy}' style='color:{color}'>{len(parts)}
                model parts for '{xy}'</h4>""")
            
            for col, model in parts.items():
                step.append(
                    self.line(f"""
                    <h6 id='step-model-{xy}-part-{col}' style='color:{color}'>
                    Select '{col}' as 'src'</h6>""")
                    )
                model_name = model['model_name']
                kwargs = model['kwargs']
                step.append(
                    self.line(f"""
                    Chooose <strong style="color:{color}">'{model_name}'
                    </strong>
                    """)
                )
                step.append(self.line("""
                Click 'Yes!', then set the kwargs to the following:"""))
                
                step.append(
                    self.line(list_group_kv(kwargs)))
                step.append(
                    self.line(f"Click 'Okay'")
                )
        step.append(self.ok("Congrats! now Click ✔️Next above"))
        return step
    
    def step_train(self) -> DOM:
        config = self.phase.config
        step = self.empty_step()
        step.append(self.title("Step 4: Training"))
        step.append("<h4>Final Step!</h4>")
        step.append("""
        🍹 You might experience some waiting while tai-chi-engine
        it's trying to download the pre-trained model""")
        step.append(self.ok(f"🚀 Click '+create' to start the training"))
        
        return step
        