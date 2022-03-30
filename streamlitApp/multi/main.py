from streamlit import cli as stcli

import streamlit as st
from multipage import MultiApp
from apps import (home, classcaus, survcaus)
import sys
sys.path.append('../')
st.set_page_config(layout="wide")


def main():
    apps = MultiApp()
    apps.add_app("Home", home.app)
    apps.add_app("ClassCaus", classcaus.app)
    apps.add_app("SurvCaus", survcaus.app)

    apps.run()

if __name__ == '__main__':
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())

