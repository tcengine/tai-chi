Tutorial
==============================================================================

Start with Dataframe
----------------------

Everything start with a dataframe, also known as `table` in python.

You can read it from a csv file::

        import pandas as pd
        df = pd.read_csv('data.csv')

Or you can read from some sql database::

        from sqlalchemy import create_engine
        with create_engine('your_db_connection_uri').connect() as conn:
            df = pd.read_sql('select * from table', conn)

Or you can read from some image directory::

        from tai_chi_engine.utils import df_creator_image_folder
        df = df_creator_image_folder('./img/directory')

It doesn't matter how, now you have a dataframe

Then you can start with::

        from tai_chi_engine import TaiChiEngine
        engine = TaiChiEngine(df, project="./example_project")
        engine()

In jupyter notebook, you will see something like this

.. image:: imgs/tai_chi_engine_start.png
    :width: 800px
    :alt: tai_chi_engine_start