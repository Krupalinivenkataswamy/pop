import streamlit as st
import pandas as pd
import itertools as itr
import numpy as np
import time
import random
from datetime import datetime,timedelta
import pydeck
import plotly.graph_objects as go
from PIL import Image


max_width = 1200
padding_top = 0
padding_right = 1
padding_left = 1
padding_bottom = 0
COLOR = "black"
BACKGROUND_COLOR = "#E8F8F5"
#BACKGROUND_COLOR = "black"

st.set_page_config(
    page_title="Crop Health",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        #Favicon  {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

heading2 = """
       <h4 style="background-color: mediumseagreen;"><b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Growing Season</b></h4>"""

heading3 = """
       <h4 style="background-color: mediumseagreen;"><b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Crop Health & Phenology</b></h4>"""

st.markdown(
    f"""
<style>
    .reportview-container .main .block-container{{
        max-width: {max_width}px;
        padding-top: {padding_top}rem;
        padding-right: {padding_right}rem;
        padding-left: {padding_left}rem;
        padding-bottom: {padding_bottom}rem;
    }}
    .reportview-container .main {{
        color: {COLOR};
        background-color: {BACKGROUND_COLOR};
    }}
</style>
""",
    unsafe_allow_html=True,
)
st.markdown(
    f'''
        <style>
            .sidebar .sidebar-content {{
                width: 260px;
            }}
        </style>
    ''',
    unsafe_allow_html=True
)

mn_path="C:/Users/Krupa/Desktop/riverbridge/Multifarm/"
vi_pth = "VI/"
vi_part = "_ndvi_linear_avg.csv"

def render_slider(date):
    key = random.random() if animation_speed else None
    date = pd.to_datetime(date, format='%d-%m-%Y')
    mn_val = date_slider.slider(
        "",
        min_value=datetime(start_date.year,start_date.month,start_date.day),
        max_value=datetime(end_date.year,end_date.month,end_date.day),
        value=datetime(date.year,date.month,date.day),
        format="DD-MM-YY",
        key=key,
    )
    date=pd.to_datetime(mn_val, format='%d-%m-%Y')
    datex = pd.to_datetime(date, format='%d-%m-%Y')
    dtx=pd.to_datetime(date, format='%d-%m-%Y').strftime(format='%d-%m-%Y')
    day = df_data.index.get_loc(datex)
    date_value.subheader(f"Date: {dtx}")
    day_value.subheader(f"Day: {day}")
    return date


def render_map(date,i):
    vgi_req = df_data.loc[[date]].T
    vgi_req.insert(1, "name", vgi_req.index)
    vgi_req.rename(
        columns={
            vgi_req.columns[0]: "vgi_req",
            vgi_req.columns[1]: "ID"
        },
        inplace=True,
    )

    df_loc["vgi"] = df_loc.merge(
        vgi_req, left_on="ID", right_on="ID"
    )["vgi_req"]

    sm_req = df_data_sm.loc[[date]].T
    sm_req.insert(1, "name", sm_req.index)
    sm_req.rename(
        columns={
            sm_req.columns[0]: "sm_req",
            sm_req.columns[1]: "ID"
        },
        inplace=True,
    )

    df_loc["sm"] = df_loc.merge(
        sm_req, left_on="ID", right_on="ID"
    )["sm_req"]

    v_mn = round(df_data.loc[date]['vi_cmp'], 6)
    v1 = df_loc["vgi"]
    v_mean = round(v1.apply(lambda x: x - v_mn), 7)
    df_loc['hlth'] = v_mean
    display_dat = df_loc[~pd.isna(df_loc["vgi"])]
    display_dat = df_loc[~pd.isna(df_loc["sm"])]

    deck_map.pydeck_chart(
        pydeck.Deck(
            map_style="mapbox://styles/mapbox/satellite-v9",
            initial_view_state=pydeck.ViewState(
                latitude=display_dat.Lat.mean(),
                longitude=display_dat.Long.mean(),
                zoom=17.5,
                pitch=54,
                bearing=i,
                height=400,
            ),
            layers=[
                pydeck.Layer(
                    "ColumnLayer",
                    data=display_dat,
                    disk_resolution=12,
                    radius=1.1,
                    elevation_scale=1,
                    get_position="[Long,Lat]",
                    get_fill_color= ['(hlth < 0 || hlth < 0.03) ? 255 : 0','(hlth < 0) ? 69 : ((hlth < 0.03) ? 215 : 128)','0'],
                    get_elevation="vgi*20",
                    auto_highlight=True,
                    pickable=True,
                ),
                pydeck.Layer(
                    "ScatterplotLayer",
                    data=display_dat,
                    opacity=0.8,
                    # stroked=True,
                    filled=True,
                    get_position="[Long,Lat]",
                    get_fill_color=['(sm <0.25)? 116 :((sm < 0.5)? 46 : 12)', '(sm <0.25)? 204 :((sm < 0.5)? 163: 113)','(sm <0.25)? 244 :((sm < 0.5)? 242: 195)'],
                    get_radius="sm*6.3",
                    #radius_scale=2,
                    auto_highlight=True,
                    pickable=True,
                ),

            ],
            tooltip={
                "html": "VI: <b>{vgi}</b></br>Hlth:<b>{hlth}</b></br>Soil Moisture: <b>{sm}</b>",
                "style": {"background": "grey", "color": "white", "font-family": '"Helvetica Neue", Arial',
                          "z-index": "10000"},
            }
        )
    )


def sow_harvest_dates(key):
    SOW_HARV_URL="C:/Users/Krupa/Desktop/riverbridge/Multifarm/calculated_sowing_harvesting_dates.csv"
    df_sow=pd.read_csv(SOW_HARV_URL)
    df_sow.set_index('Crop_ID', inplace=True, drop=True)
    df_sow['Calculated_sowing'] = pd.to_datetime(df_sow['Calculated_sowing'], format='%d-%m-%Y')
    df_sow['calculated_harvest'] = pd.to_datetime(df_sow['calculated_harvest'], format='%d-%m-%Y')
    sh_dates=df_sow.loc[[key]]
    sh_dates.reset_index(drop=True,inplace=True)
    st=sh_dates['Calculated_sowing']
    ed=sh_dates['calculated_harvest']
    start_date=pd.to_datetime(st[0], format='%d-%m-%Y')
    end_date=pd.to_datetime(ed[0], format='%d-%m-%Y')
    #start_date_f=pd.to_datetime(st[0], format='%d-%m-%Y').strftime(format='%d-%m-%Y')
    #end_date_f=pd.to_datetime(ed[0], format='%d-%m-%Y').strftime(format='%d-%m-%Y')
    begin_dt = datetime(2020, start_date.month, start_date.day)
    return start_date,end_date,begin_dt

def kpi_ind(date):
    soil = df_data_sm.loc[date]
    temp_dt=df_aw.loc[[date]]
    temp_max = int(temp_dt['MaxTemp_C'])
    temp_min = int(temp_dt['MinTemp_C'])
    hum = int(temp_dt['Relative_Humd'])
    preci = int(temp_dt['precipitation'])
    temp = int(temp_dt['Temp_C'])
    datex = pd.to_datetime(date, format='%d-%m-%Y')
    mask_del = (df_avg_farm.index.month == datex.month) & (df_avg_farm.index.day == datex.day)
    df_avg_req = df_avg_farm[mask_del]
    temp_del = df_avg_req['hi_temp']
    hum_del = df_avg_req['hi_relative_humd']
    preci_del = df_avg_req['hi_precipitation']

    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="gauge+delta",
        value=temp,
        number={'suffix': "°C", 'font': {'size': 20}},
        domain={'x': [0, 0], 'y': [0, 0]},
        title={'text': "Temperature", 'font': {'size': 20}},
        delta={'reference': temp_del[0]},
        gauge={
            'axis': {'range': [0,55], 'tickwidth': 2, 'tickcolor': "black"},
            'bar': {'color': "firebrick"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "black",
            'steps': [
                {'range': [0, 55], 'color': 'lightgray'},
                {'range': [temp_min, temp_max], 'color': 'sandybrown'},
            ]
        }))
    fig.add_trace(go.Indicator(
        mode="gauge+delta",
        value=hum,
        number={'suffix': "RH", 'font': {'size': 20}},
        domain={'row': 0, 'column': 1},
        title={'text': "Humidity", 'font': {'size': 20}},
        delta={'reference': hum_del[0]},
        gauge={
            'axis': {'range': [0, 99], 'tickwidth': 2, 'tickcolor': "black"},
            'bar': {'color': "teal"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "black",
            'steps': [
                {'range': [0, 50], 'color': 'lightgray'},
                {'range': [50, 99], 'color': 'darkgray'},
            ]
        }))
    fig.add_trace(go.Indicator(
        mode="gauge+delta",
        value=soil['kpi_avg'],
        number={'suffix': "RH", 'font': {'size': 20}},
        domain={'row': 0, 'column': 2},
        title={'text': "Soil Moisture", 'font': {'size': 20}},
        delta={'reference': 17},
        gauge={
            'axis': {'range': [-2, 2], 'tickwidth': 2, 'tickcolor': "black"},
            'bar': {'color': "gold"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "black",
            'steps': [
                {'range': [-2, 0], 'color': 'lightgray'},
                {'range': [0, 2], 'color': 'darkgray'},
            ]
        }))

    fig.add_trace(go.Indicator(
        mode="gauge+delta",
        value=preci,
        number={'suffix': "mm", 'font': {'size': 20}},
        domain={'row': 1, 'column': 3},
        title={'text': "Precipitation", 'font': {'size': 20}},
        delta={'reference': preci_del[0]},
        gauge={
            'axis': {'range': [-5, 10], 'tickwidth': 2, 'tickcolor': "black"},
            'bar': {'color': "cornflowerblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "black",
            'steps': [
                {'range': [-5, 0], 'color': 'lightgray'},
                {'range': [0, 10], 'color': 'darkgray'},
            ]
        }))

    fig.update_layout(
        grid={'rows': 1, 'columns': 4, 'pattern': "independent"},
        margin={'t': 35, 'l': 15, 'r': 15, 'b': 0},
        paper_bgcolor="aliceblue",
        autosize=True,
        # width=850,
        height=255,
        template={'data': {'indicator': [{
            'title': {'text': "KPIs"},
            'mode': "gauge+delta",
            'delta': {'reference': 90}}]
        }},
        font={'color': "darkblue", 'family': "Arial"},
    )
    k_plot.plotly_chart(fig ,use_container_width=True)


def kpi_prep_data(key):
    AW_URLS = {
        2802: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/thermal_unit_avg_v4/thermal_unit_std_cropid-2802.csv",
        2803: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/thermal_unit_avg_v4/thermal_unit_std_cropid-2803.csv",
        2804: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/thermal_unit_avg_v4/thermal_unit_std_cropid-2804.csv",
        2806: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/thermal_unit_avg_v4/thermal_unit_std_cropid-2806.csv",
        2807: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/thermal_unit_avg_v4/thermal_unit_std_cropid-2807.csv",
        2808: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/thermal_unit_avg_v4/thermal_unit_std_cropid-2808.csv",
        2810: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/thermal_unit_avg_v4/thermal_unit_std_cropid-2810.csv",
        2811: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/thermal_unit_avg_v4/thermal_unit_std_cropid-2811.csv",
        2812: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/thermal_unit_avg_v4/thermal_unit_std_cropid-2812.csv",
        2813: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/thermal_unit_avg_v4/thermal_unit_std_cropid-2813.csv",
        2814: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/thermal_unit_avg_v4/thermal_unit_std_cropid-2814.csv",
        2815: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/thermal_unit_avg_v4/thermal_unit_std_cropid-2815.csv",
        2816: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/thermal_unit_avg_v4/thermal_unit_std_cropid-2816.csv",
        2817: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/thermal_unit_avg_v4/thermal_unit_std_cropid-2817.csv",
        2818: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/thermal_unit_avg_v4/thermal_unit_std_cropid-2818.csv",
        2819: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/thermal_unit_avg_v4/thermal_unit_std_cropid-2819.csv",
        2820: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/thermal_unit_avg_v4/thermal_unit_std_cropid-2820.csv",
        2823: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/thermal_unit_avg_v4/thermal_unit_std_cropid-2823.csv",
        2824: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/thermal_unit_avg_v4/thermal_unit_std_cropid-2824.csv",
        2825: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/thermal_unit_avg_v4/thermal_unit_std_cropid-2825.csv",
        2826: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/thermal_unit_avg_v4/thermal_unit_std_cropid-2826.csv",
        2827: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/thermal_unit_avg_v4/thermal_unit_std_cropid-2827.csv",
        2829: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/thermal_unit_avg_v4/thermal_unit_std_cropid-2829.csv",
        2830: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/thermal_unit_avg_v4/thermal_unit_std_cropid-2830.csv"
    }

    df_aw = pd.read_csv(AW_URLS[key], index_col=None)
    df_aw.set_index('Date', inplace=True, drop=True)
    df_aw.index = pd.to_datetime(df_aw.index, format='%d-%m-%Y')

    AVGDATA = "C:/Users/Krupa/Desktop/riverbridge/Multifarm/combined/tbl_avg_weather_historical_data_26fields_29102020.csv"
    df_avg = pd.read_csv(AVGDATA)
    avg_mask = (df_avg['hi_name'] == key)
    df_avg_farm = df_avg[avg_mask]
    df_avg_farm.set_index('hi_date', inplace=True, drop=True)
    df_avg_farm.index = pd.to_datetime(df_avg_farm.index, format='%d-%m-%Y')
    # prep for gdd and other plots
    # average temp as per sowing dates
    mask_avg1 = (df_avg_farm.index >= begin_dt)
    df_avg1 = df_avg_farm.loc[mask_avg1]
    mask_avg2 = (df_avg_farm.index <= end_date)
    df_avg2 = df_avg_farm.loc[mask_avg2]
    df_avg_comb = df_avg1.append(df_avg2)
    # df_avg_comb.reset_index(drop=True,inplace=True)
    len_df = len(df_avg_comb)
    df_avg_comb['day_no'] = np.arange(1, len_df + 1, 1)
    # actual temp as per sowing dates
    df_aw_cp = df_aw.copy()
    len_aw = len(df_aw_cp)
    df_aw_cp['day_no'] = np.arange(1, len_aw + 1, 1)
    df_aw_cp.reset_index(drop=True, inplace=True)
    return df_aw, df_avg_farm,df_aw_cp,df_avg_comb

def get_data(key,ckey):
    mn_path = "C:/Users/Krupa/Desktop/riverbridge/Multifarm/"
    ndvi = "ndvi/"
    ndvi_part = "_Popcorn.csv"


    SM_URLS = {
    2802: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/Soil_moisture_clip/2802_Popcorn Hybrid GP-206_soil_moisture.csv",
    2803: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/Soil_moisture_clip/2803_Popcorn Hybrid GP-206_soil_moisture.csv",
    2804: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/Soil_moisture_clip/2804_Popcorn Hybrid GP-206_soil_moisture.csv",
    2806: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/Soil_moisture_clip/2806_Popcorn Hybrid GP-208_soil_moisture.csv",
    2807: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/Soil_moisture_clip/2807_Popcorn Hybrid GP-208_soil_moisture.csv",
    2808: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/Soil_moisture_clip/2808_Popcorn Hybrid GP-208_soil_moisture.csv",
    2810: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/Soil_moisture_clip/2810_Popcorn Hybrid GP-208_soil_moisture.csv",
    2811: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/Soil_moisture_clip/2811_Popcorn Hybrid GP-208_soil_moisture.csv",
    2812: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/Soil_moisture_clip/2812_Popcorn Hybrid GP-208_soil_moisture.csv",
    2813: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/Soil_moisture_clip/2813_Popcorn Hybrid GP-208_soil_moisture.csv",
    2814: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/Soil_moisture_clip/2814_Popcorn Hybrid GP-208_soil_moisture.csv",
    2815: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/Soil_moisture_clip/2815_Maize_soil_moisture.csv",
    2816: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/Soil_moisture_clip/2816_Maize_soil_moisture.csv",
    2817: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/Soil_moisture_clip/2817_Popcorn Hybrid GP-206_soil_moisture.csv",
    2818: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/Soil_moisture_clip/2818_Popcorn Hybrid GP-208_soil_moisture.csv",
    2819: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/Soil_moisture_clip/2819_Popcorn Hybrid GP-208_soil_moisture.csv",
    2820: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/Soil_moisture_clip/2820_Popcorn Hybrid GP-208_soil_moisture.csv",
    2823: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/Soil_moisture_clip/2823_Popcorn Hybrid GP-206_soil_moisture.csv",
    2824: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/Soil_moisture_clip/2824_Popcorn Hybrid GP-206_soil_moisture.csv",
    2825: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/Soil_moisture_clip/2825_Maize_soil_moisture.csv",
    2826: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/Soil_moisture_clip/2826_Maize_soil_moisture.csv",
    2827: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/Soil_moisture_clip/2827_Popcorn Hybrid GP-208_soil_moisture.csv",
    2829: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/Soil_moisture_clip/2829_Popcorn Hybrid GP-208_soil_moisture.csv",
    2830: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/Soil_moisture_clip/2830_Popcorn Hybrid GP-208_soil_moisture.csv"
    }

    ALL_URL = "C:/Users/Krupa/Desktop/riverbridge/Multifarm/VI/all.csv"
    df_all = pd.read_csv(ALL_URL, index_col=False)
    df_full = pd.read_csv(f"{mn_path}{ndvi}{fkey}{ndvi_part}", index_col=False, header=None)


    df_lat = df_full.loc[[0, 1], :].T.reset_index()
    df_dat = df_full.drop([0, 1])  # dropping Lat and Long rows
    df_lat.insert(3, "ID", df_lat.index + 100)
    new_header = df_lat.iloc[0]
    df_lat = df_lat[1:]
    df_lat.columns = new_header
    df_lat.rename(
        columns={
            df_lat.columns[3]: "ID",
        },
        inplace=True,
    )
    df_dat.columns = range(100, len(df_dat.columns) + 100)
    df_dat_new = df_dat.rename(columns={df_dat.columns[0]: 'date'})
    df_dat_new.set_index('date', inplace=True, drop=True)
    df_dat_new.index = pd.to_datetime(df_dat_new.index, format='%d-%m-%Y')
    mask_main = (df_dat_new.index >= start_date) & (df_dat_new.index <= end_date)
    df_dat_req = df_dat_new.loc[mask_main]
    # df_dat_req.index = pd.to_datetime(df_dat_req.index, format='%d-%m-%Y').strftime('%d-%m-%Y')
    # df_dat_req.reset_index(drop=True,inplace=True)
    df_lat.drop(df_lat.columns[0], axis=1, inplace=True)

    #comparison of health
    if(ckey=='All'):
        df_all['vi_mn'] = df_all.mean(axis=1)
        len_aq = len(df_all)
        df_all['day_no'] = np.arange(1, len_aq + 1, 1)
        len_ac=len(df_dat_req)
        df_dat_all_new = df_all.iloc[0:len_ac]
        vi_add = df_dat_all_new['vi_mn']
        vi_add.reset_index(drop=True, inplace=True)
        df_va = pd.DataFrame(vi_add)
        df_va['day_no'] = np.arange(0, len_ac,1)
        df_dat_req['day_no'] = np.arange(0, len_ac,1)
        if (len_aq < len_ac):
            diff_len = len_ac - len_aq
            app_lst = np.repeat(df_dat_all[len_cmp - 1]['vi_mn'], diff_len)
            vi_add.append(app_lst)
        df_dat_req['vi_cmp'] = df_dat_req['day_no'].map(vi_add)

    else:
        df_full_c = pd.read_csv(f"{mn_path}{ndvi}{ckey}{ndvi_part}",index_col=False,header=None)
        df_dat_c = df_full_c.drop([0, 1])  # dropping Lat and Long rows
        start_date_c,end_date_c,begin_dt_c=sow_harvest_dates(ckey)
        df_dat_c.columns = range(100, len(df_dat_c.columns) + 100)
        df_dat_cn = df_dat_c.rename(columns={df_dat_c.columns[0]: 'date'})
        df_dat_cn.set_index('date', inplace=True, drop=True)
        df_dat_cn.index = pd.to_datetime(df_dat_cn.index, format='%d-%m-%Y')
        mask_main_c = (df_dat_cn.index >= start_date_c)
        df_dat_cr = df_dat_cn.loc[mask_main_c]
        df_dat_cr['vi_mn']=df_dat_cr.mean(axis=1)
        len_ac =len(df_dat_req)
        len_cmp=len(df_dat_cr)
        df_dat_cr_new=df_dat_cr.iloc[0:len_ac]
        #end_date_new= end_date + timedelta(1)
        #lert=np.arange(start_date,end_date_new,dtype='datetime64[D]')
        #df_dat_cr_new['date']=np.arange(start_date,end_date)
        vi_add=df_dat_cr_new['vi_mn']
        vi_add.reset_index(drop=True,inplace=True)
        df_va=pd.DataFrame(vi_add)
        df_va['day_no']=np.arange(0,len_ac,1)
        df_dat_req['day_no']=np.arange(0,len_ac,1)
        if(len_cmp < len_ac):
            diff_len=len_ac-len_cmp
            app_lst=np.repeat(df_dat_cr[len_cmp-1]['vi_mn'],diff_len)
            vi_add.append(app_lst)
        df_dat_req['vi_cmp'] = df_dat_req['day_no'].map(vi_add)

    df_dk_sm = pd.read_csv(SM_URLS[key],index_col=False,header=None)
    df_lat_sm = df_dk_sm.loc[[0, 1], :].T.reset_index()
    df_dat_sm = df_dk_sm.drop([0, 1])  # dropping Lat and Long rows
    df_lat_sm.insert(3, "ID", df_lat_sm.index + 100)
    new_header_sm = df_lat_sm.iloc[0]
    df_lat_sm = df_lat_sm[1:]
    df_lat_sm.columns = new_header_sm
    df_lat_sm.rename(
        columns={
            df_lat_sm.columns[3]: "ID",
        },
        inplace=True,
    )
    df_dat_sm.columns = range(100, len(df_dat_sm.columns) + 100)
    df_dat_new_sm = df_dat_sm.rename(columns={df_dat_sm.columns[0]: 'date'})
    df_dat_new_sm.set_index('date', inplace=True, drop=True)
    df_dat_new_sm.index = pd.to_datetime(df_dat_new_sm.index, format='%d-%m-%Y')
    df_dat_new_sm['kpi_avg']=df_dat_new_sm.mean(axis=1)
    #mask_main_sm = (df_dat_new_sm.index >= start_date) & (df_dat_new_sm.index <= end_date)
    #df_dat_req_sm = df_dat_new_sm.loc[mask_main_sm]
    #df_dat_req_sm.index = pd.to_datetime(df_dat_req_sm.index, format='%d-%m-%Y').strftime('%d-%m-%Y')
    #df_dat_new_sm.reset_index(drop=True,inplace=True)
    df_lat_sm.drop(df_lat_sm.columns[0], axis=1, inplace=True)
    return df_lat, df_dat_req, df_lat_sm, df_dat_new_sm

def gdd():
    avg_temp = ((df_avg_comb['hi_max_temp'] + df_avg_comb['hi_min_temp']) / 2) - 10
    avg_temp_cum = avg_temp.cumsum()
    df_avg_comb['temp_cu'] = avg_temp_cum
    gdd = df_aw_cp['ThermalTemp_C']
    chu = gdd.cumsum()
    df_aw_cp['chu'] = chu
    x_axis = df_aw_cp.index
    x_len = len(df_aw_cp)
    xlist = [*range(1, x_len + 1, 1)]
    df_aw_cp['day_no'] = xlist
    y_axis = df_aw_cp["chu"]
    mask_z = df_aw_cp["precipitation"] > 0
    z = df_aw_cp.loc[mask_z]
    z_axis = z["chu"]
    z_date = z["day_no"]

    # avg_precipitation
    # df_avg_comb['day_no'] = xlist
    mask_avg_p = df_avg_comb['hi_precipitation'] > 0
    avg_req = df_avg_comb.loc[mask_avg_p]
    avg_yaxis = avg_req['temp_cu']
    avg_p_date = avg_req['day_no']

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xlist, y=avg_temp_cum,
                             mode='lines',
                             name='avg GDUs'))
    fig.add_trace(go.Scatter(x=avg_p_date, y=avg_yaxis,
                             mode='markers',
                             name='avg_precipitation',
                             marker=dict(color='peru', size=6)))
    fig.add_trace(go.Scatter(x=xlist, y=y_axis,
                             mode='lines',
                             name='Growing season'))
    fig.add_trace(go.Scatter(x=z_date, y=z_axis,
                             mode='markers',
                             name='precipitation',
                             marker=dict(color='red', size=6)))
    fig.add_trace(go.Scatter(x=xlist, y=gdd*gdd,
                             mode='lines',
                             name='gdd',
                             marker=dict(color='indigo')))
    fig.add_trace(go.Scatter(x=xlist, y=avg_temp*avg_temp,
                             mode='lines',
                             name='avg_gdd',
                             marker=dict(color='darkorange')))
    fig.update_layout(
        margin={'t': 30, 'l': 5, 'r': 5, 'b': 10},
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        paper_bgcolor="white",
        width=530,
        height=320,
        title_text="Thermal Units",
        title_x=0.5,
        font={'color': "darkblue", 'family': "Arial"},
    )
    gdd_plot.plotly_chart(fig,use_container_width=True)

def ph_base(key,ckey,date):
    PH_URLS = {
        2802: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_Avg_New/2802_DPD_linear_avg.csv",
        2803: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_Avg_New/2803_DPD_linear_avg.csv",
        2804: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_Avg_New/2804_DPD_linear_avg.csv",
        2806: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_Avg_New/2806_DPD_linear_avg.csv",
        2807: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_Avg_New/2807_DPD_linear_avg.csv",
        2808: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_Avg_New/2808_DPD_linear_avg.csv",
        2810: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_Avg_New/2810_DPD_linear_avg.csv",
        2811: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_Avg_New/2811_DPD_linear_avg.csv",
        2812: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_Avg_New/2812_DPD_linear_avg.csv",
        2813: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_Avg_New/2813_DPD_linear_avg.csv",
        2814: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_Avg_New/2814_DPD_linear_avg.csv",
        2815: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_Avg_New/2815_DPD_linear_avg.csv",
        2816: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_Avg_New/2816_DPD_linear_avg.csv",
        2817: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_Avg_New/2817_DPD_linear_avg.csv",
        2818: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_Avg_New/2818_DPD_linear_avg.csv",
        2819: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_Avg_New/2819_DPD_linear_avg.csv",
        2820: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_Avg_New/2820_DPD_linear_avg.csv",
        2823: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_Avg_New/2823_DPD_linear_avg.csv",
        2824: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_Avg_New/2824_DPD_linear_avg.csv",
        2825: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_Avg_New/2825_DPD_linear_avg.csv",
        2826: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_Avg_New/2826_DPD_linear_avg.csv",
        2827: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_Avg_New/2827_DPD_linear_avg.csv",
        2829: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_Avg_New/2829_DPD_linear_avg.csv",
        2830: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_Avg_New/2830_DPD_linear_avg.csv"
    }

    REF_URLS="C:/Users/Krupa/Desktop/riverbridge/Multifarm/GP_PhenoStages_Values.csv"
    df_ref=pd.read_csv(REF_URLS,index_col=False)

    ph="phenology/"
    ph_part="_dpd_plant.csv"
    df_dpd=pd.read_csv(f"{mn_path}{ph}{fkey}{ph_part}", index_col=False)
    len_dp=len(df_dpd)
    df_dpd['day_no']=np.arange(0,len_dp,1)



    mask_ref=(df_ref['crop_id']==key)
    df_ref_req=df_ref.loc[mask_ref]
    ve=df_ref_req['VE'].values[0]
    df_ve=[ve,ve]

    v6=df_ref_req['V6'].values[0]
    df_v6=[v6,v6]

    v12=df_ref_req['V12'].values[0]
    df_v12=[v12,v12]

    vt=df_ref_req['VT'].values[0]
    df_vt=[vt,vt]

    r1=df_ref_req['R1'].values[0]
    df_r1=[r1,r1]

    r3=df_ref_req['R3'].values[0]
    df_r3=[r3,r3]

    r5=df_ref_req['R5'].values[0]
    df_r5=[r5,r5]

    r6=df_ref_req['R6'].values[0]
    df_r6=[r6,r6]

    df_ph = pd.read_csv(PH_URLS[key], index_col=None)
    len1 = len(df_ph)
    df_ph.rename(
        columns={
            df_ph.columns[1]: "ph",
        },
        inplace = True,
    )
    df_ph['day_no'] = np.arange(1, len1 + 1, 1)
    ph_axis=df_ph['day_no']
    ph=df_ph['ph']
    ymax=ph.max()

    df_ph_cmp = pd.read_csv(PH_URLS[ckey], index_col=None)
    len1 = len(df_ph_cmp)
    df_ph_cmp.rename(
        columns={
            df_ph_cmp.columns[1]: "ph",
        },
        inplace=True,
    )
    df_ph_cmp['day_no'] = np.arange(1, len1 + 1, 1)
    phc_axis = df_ph['day_no']
    phc = df_ph_cmp['ph']


    bl1=df_data.loc[date]
    bl2=int(bl1['day_no'])
    df_bl = df_dpd.loc[bl2]
    bl_x=[df_bl['gdd'],df_bl['gdd']]
    bl_y=[0,ymax]


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=phc_axis, y=phc,
                                     mode='lines',
                                     name='Benchmark',
                             line=dict(color='teal') ))
    fig.add_trace(go.Scatter(x=ph_axis, y=ph,
                                     mode='lines',
                                     name='Current',
                             line=dict(color='mediumblue')))
    fig.add_trace(go.Scatter(x=bl_x, y=bl_y,
                                     mode='lines',
                                     showlegend=False,
                             line=dict(color='black')))
    fig.add_trace(go.Scatter(x=df_ve, y=[0,ymax],
                                     mode='lines',
                                     showlegend=False,
                             line = dict(color='slategray', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=df_v6, y=[0,ymax],
                                     mode='lines',
                                     showlegend=False,
                             line = dict(color='slategray', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=df_v12, y=[0,ymax],
                                     mode='lines',
                                     showlegend=False,
                             line = dict(color='slategray', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=df_vt, y=[0,ymax],
                                     mode='lines',
                                     showlegend=False,
                             line = dict(color='slategray', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=df_r1, y=[0,ymax],
                                     mode='lines',
                                     showlegend=False,
                             line = dict(color='slategray', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=df_r3, y=[0,ymax],
                                     mode='lines',
                                     showlegend=False,
                             line = dict(color='slategray', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=df_r5, y=[0,ymax],
                                     mode='lines',
                                     showlegend=False,
                             line = dict(color='slategray', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=df_r6, y=[0,ymax],
                                     mode='lines',
                                     showlegend=False,
                             line = dict(color='slategray', width=2, dash='dash')))
    fig.add_annotation(
        x=ve,  # arrows' head
        y=0.1,  # arrows' head
        ax=vt,
        ay=0.1,
        xref='x',
        yref='y',
        axref='x',
        ayref='y',
        text='',
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor='peru'
    )
    fig.add_annotation(
        x=vt,  # arrows' head
        y=0.1,  # arrows' head
        ax=ve,
        ay=0.1,
        xref='x',
        yref='y',
        axref='x',
        ayref='y',
        text='',  # if you want only the arrow
        align='center',
        yanchor='bottom',
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor='peru'
    )
    fig.add_annotation(x=(ve+vt)/2, y=0.1,
                       text="<b>Vegetative Stage</b>",
                       showarrow=False,
                       yshift=10)

    fig.add_annotation(
        x=r1,  # arrows' head
        y=0.1,  # arrows' head
        ax=r6,
        ay=0.1,
        xref='x',
        yref='y',
        axref='x',
        ayref='y',
        text='',  # if you want only the arrow
        align='center',
        yanchor='bottom',
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor='teal'
    )
    fig.add_annotation(
        x=r6,  # arrows' head
        y=0.1,  # arrows' head
        ax=r1,
        ay=0.1,
        xref='x',
        yref='y',
        axref='x',
        ayref='y',
        text='',
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor='teal'
    )
    fig.add_annotation(x=(r1+r6)/2, y=0.1,
                       text="<b>Reproductive Stage</b>",
                       showarrow=False,
                       yshift=10)

    fig.update_layout(
                xaxis=dict(
                    tickmode='array',
                    tickvals=[115,555,945,1350,1400,1925,2450,2700],
                    ticktext=['VE','V6', 'V12', 'VT', 'R1','R3','R5', 'R6'],
                    tickangle=55,
                    tickfont=dict(size=11)),
                margin={'t': 0, 'l': 5, 'r': 5, 'b': 10},
                legend=dict(yanchor="top", y=0.97, xanchor="left", x=0.02),
                paper_bgcolor="#E8F8F5",
                width=750,
                height=325,
                # title_text="Corn Phenology",
                # title_x=0.5,
                font={'color': "darkblue", 'family': "Arial"},
    )
    ph_plot.plotly_chart(fig,use_column_width=True)

def plant(date):
 dummy= Image.open("C:/Users/Krupa/Desktop/riverbridge/2804/Images02/dummy.png")
 ve_image=Image.open("C:/Users/Krupa/Desktop/riverbridge/2804/Images02/AGB-Image-edit-VEN.png")
 v1_image = Image.open("C:/Users/Krupa/Desktop/riverbridge/2804/Images02/AGB-Image-edit-V1N.png")
 v3_image = Image.open("C:/Users/Krupa/Desktop/riverbridge/2804/Images02/AGB-Image-edit-V3N.png")
 v6_image = Image.open("C:/Users/Krupa/Desktop/riverbridge/2804/Images02/AGB-Image-edit-V6N.png")
 v9_image = Image.open("C:/Users/Krupa/Desktop/riverbridge/2804/Images02/AGB-Image-edit-V9N.png")
 vt_image = Image.open("C:/Users/Krupa/Desktop/riverbridge/2804/Images02/AGB-Image-edit-VTN.png")
 R1_image = Image.open("C:/Users/Krupa/Desktop/riverbridge/2804/Images02/AGB-Image-edit-R1N.png")
 R2_image = Image.open("C:/Users/Krupa/Desktop/riverbridge/2804/Images02/AGB-Image-edit-R2.png")
 R3_image = Image.open("C:/Users/Krupa/Desktop/riverbridge/2804/Images02/AGB-Image-edit-R3.png")
 R4_image = Image.open("C:/Users/Krupa/Desktop/riverbridge/2804/Images02/AGB-Image-edit-R4.png")
 R5_image = Image.open("C:/Users/Krupa/Desktop/riverbridge/2804/Images02/AGB-Image-edit-R5.png")
 R6_image = Image.open("C:/Users/Krupa/Desktop/riverbridge/2804/Images02/AGB-Image-edit-R6N.png")


 datex = pd.to_datetime(date, format='%d-%m-%Y')
 gdd=df_aw['ThermalTemp_F']
 chu = gdd.cumsum()
 df_aw['chu'] = chu
 gdd_pl = df_aw.loc[datex]['chu']
  #pl_chu=df_day['chu']
 PLNT_URLS = "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenostages_plant.csv"
 df_plnt = pd.read_csv(PLNT_URLS, index_col=False)
 vf=df_plnt.iloc[0]
 #tt=df_aw.loc[end_date]['chu']


 #(VE,V1,V3,V6,V12,VT,R1,R2,R3,R4,R5,R6)=(110, 180, 350, 610, 740, 1135, 1250, 1660, 1830, 1925, 2450, 2700)


 if (gdd_pl < vf['VE']):
  image_crn.image(dummy, width=10, caption="")
 elif (gdd_pl == vf['VE'] and gdd_pl < vf['V1']):
  image_crn.image(ve_image,width=100,caption="VE Stage")
 elif (gdd_pl >= vf['V1'] and gdd_pl < vf['V3']):
   image_crn.image(v1_image,width=100,caption="V1 Stage")
 elif (gdd_pl >= vf['V3'] and gdd_pl < vf['V6']):
   image_crn.image(v3_image,width=100,caption="V3 Stage")
 elif (gdd_pl >= vf['V6'] and gdd_pl < vf['V12']):
   image_crn.image(v6_image, width=100,caption="V6 Stage")
 elif (gdd_pl >= vf['V12'] and gdd_pl < vf['VT']):
   image_crn.image(v9_image, caption="V12 Stage")
 elif (gdd_pl >= vf['VT'] and gdd_pl < vf['R1']):
   image_crn.image(vt_image,caption="VT Stage")
 elif (gdd_pl >= vf['R1'] and gdd_pl < vf['R2']):
   image_crn.image(R1_image,caption="R1 Stage")
 elif (gdd_pl >= vf['R2'] and gdd_pl < vf['R3']):
   image_crn.image(R2_image,width=100,caption="R2 Stage")
 elif (gdd_pl >= vf['R3'] and gdd_pl < vf['R4']):
   image_crn.image(R3_image,width=100,caption="R3 Stage")
 elif (gdd_pl >= vf['R4'] and gdd_pl <  vf['R5']):
   image_crn.image(R4_image, width=100,caption="R4 Stage")
 elif (gdd_pl >= vf['R5'] and gdd_pl < vf['R6']):
   image_crn.image(R5_image,width=100, caption="R5 Stage")
 elif (gdd_pl >= vf['R6']):
   image_crn.image(R6_image, caption="R6 Stage")


def crp(key):
 PH_ND_URLS = {
        2802: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_scale_Avg_New/2802_DPD_linear_scale_avg.csv",
        2803: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_scale_Avg_New/2803_DPD_linear_scale_avg.csv",
        2804: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_scale_Avg_New/2804_DPD_linear_scale_avg.csv",
        2806: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_scale_Avg_New/2806_DPD_linear_scale_avg.csv",
        2807: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_scale_Avg_New/2807_DPD_linear_scale_avg.csv",
        2808: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_scale_Avg_New/2808_DPD_linear_scale_avg.csv",
        2810: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_scale_Avg_New/2810_DPD_linear_scale_avg.csv",
        2811: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_scale_Avg_New/2811_DPD_linear_scale_avg.csv",
        2812: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_scale_Avg_New/2812_DPD_linear_scale_avg.csv",
        2813: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_scale_Avg_New/2813_DPD_linear_scale_avg.csv",
        2814: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_scale_Avg_New/2814_DPD_linear_scale_avg.csv",
        2815: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_scale_Avg_New/2815_DPD_linear_scale_avg.csv",
        2816: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_scale_Avg_New/2816_DPD_linear_scale_avg.csv",
        2817: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_scale_Avg_New/2817_DPD_linear_scale_avg.csv",
        2818: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_scale_Avg_New/2818_DPD_linear_scale_avg.csv",
        2819: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_scale_Avg_New/2819_DPD_linear_scale_avg.csv",
        2820: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_scale_Avg_New/2820_DPD_linear_scale_avg.csv",
        2823: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_scale_Avg_New/2823_DPD_linear_scale_avg.csv",
        2824: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_scale_Avg_New/2824_DPD_linear_scale_avg.csv",
        2825: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_scale_Avg_New/2825_DPD_linear_scale_avg.csv",
        2826: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_scale_Avg_New/2826_DPD_linear_scale_avg.csv",
        2827: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_scale_Avg_New/2827_DPD_linear_scale_avg.csv",
        2829: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_scale_Avg_New/2829_DPD_linear_scale_avg.csv",
        2830: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/NDVI_Linear_scale_Avg_New/2830_DPD_linear_scale_avg.csv"
 }

 PH_VG_URLS = {
     2802: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/Vari_Green_Linear_scale_Avg_New/2802_Vari_Green_DPD_linear_scale_avg.csv",
     2803: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/Vari_Green_Linear_scale_Avg_New/2803_Vari_Green_DPD_linear_scale_avg.csv",
     2804: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/Vari_Green_Linear_scale_Avg_New/2804_Vari_Green_DPD_linear_scale_avg.csv",
     2806: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/Vari_Green_Linear_scale_Avg_New/2806_Vari_Green_DPD_linear_scale_avg.csv",
     2807: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/Vari_Green_Linear_scale_Avg_New/2807_Vari_Green_DPD_linear_scale_avg.csv",
     2808: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/Vari_Green_Linear_scale_Avg_New/2808_Vari_Green_DPD_linear_scale_avg.csv",
     2810: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/Vari_Green_Linear_scale_Avg_New/2810_Vari_Green_DPD_linear_scale_avg.csv",
     2811: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/Vari_Green_Linear_scale_Avg_New/2811_Vari_Green_DPD_linear_scale_avg.csv",
     2812: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/Vari_Green_Linear_scale_Avg_New/2812_Vari_Green_DPD_linear_scale_avg.csv",
     2813: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/Vari_Green_Linear_scale_Avg_New/2813_Vari_Green_DPD_linear_scale_avg.csv",
     2814: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/Vari_Green_Linear_scale_Avg_New/2814_Vari_Green_DPD_linear_scale_avg.csv",
     2815: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/Vari_Green_Linear_scale_Avg_New/2815_Vari_Green_DPD_linear_scale_avg.csv",
     2816: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/Vari_Green_Linear_scale_Avg_New/2816_Vari_Green_DPD_linear_scale_avg.csv",
     2817: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/Vari_Green_Linear_scale_Avg_New/2817_Vari_Green_DPD_linear_scale_avg.csv",
     2818: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/Vari_Green_Linear_scale_Avg_New/2818_Vari_Green_DPD_linear_scale_avg.csv",
     2819: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/Vari_Green_Linear_scale_Avg_New/2819_Vari_Green_DPD_linear_scale_avg.csv",
     2820: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/Vari_Green_Linear_scale_Avg_New/2820_Vari_Green_DPD_linear_scale_avg.csv",
     2823: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/Vari_Green_Linear_scale_Avg_New/2823_Vari_Green_DPD_linear_scale_avg.csv",
     2824: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/Vari_Green_Linear_scale_Avg_New/2824_Vari_Green_DPD_linear_scale_avg.csv",
     2825: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/Vari_Green_Linear_scale_Avg_New/2825_Vari_Green_DPD_linear_scale_avg.csv",
     2826: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/Vari_Green_Linear_scale_Avg_New/2826_Vari_Green_DPD_linear_scale_avg.csv",
     2827: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/Vari_Green_Linear_scale_Avg_New/2827_Vari_Green_DPD_linear_scale_avg.csv",
     2829: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/Vari_Green_Linear_scale_Avg_New/2829_Vari_Green_DPD_linear_scale_avg.csv",
     2830: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/phenology/Vari_Green_Linear_scale_Avg_New/2830_Vari_Green_DPD_linear_scale_avg.csv"
 }

 df_ph_nd = pd.read_csv(PH_ND_URLS[key], index_col=None)
 len2 = len(df_ph_nd)
 df_ph_nd.rename(
     columns={
         df_ph_nd.columns[1]: "ph_nd",
     },
     inplace = True,
 )
 df_ph_nd['day_no'] = np.arange(1, len2 + 1, 1)
 df_ph_vg = pd.read_csv(PH_VG_URLS[key], index_col=None)
 len3 = len(df_ph_vg)
 df_ph_vg.rename(
     columns={
         df_ph_vg.columns[1]: "ph_vg",
     },
     inplace = True,
 )
 df_ph_vg['day_no'] = np.arange(1, len3 + 1, 1)

 x_axis = df_ph_nd['day_no']
 nd=df_ph_nd['ph_nd']
 x_axis_vg= df_ph_vg['day_no']
 vg=(df_ph_vg['ph_vg'] * -1)
 vgmax=vg.max()

 REF_URLS = "C:/Users/Krupa/Desktop/riverbridge/Multifarm/GP_PhenoStages_Values.csv"
 df_ref = pd.read_csv(REF_URLS, index_col=False)

 mask_ref = (df_ref['crop_id'] == key)
 df_ref_req = df_ref.loc[mask_ref]
 ve = df_ref_req['VE'].values[0]
 df_ve = [ve, ve]

 v6 = df_ref_req['V6'].values[0]
 df_v6 = [v6, v6]

 v12 = df_ref_req['V12'].values[0]
 df_v12 = [v12, v12]

 vt = df_ref_req['VT'].values[0]
 df_vt = [vt, vt]

 r1 = df_ref_req['R1'].values[0]
 df_r1 = [r1, r1]

 r3 = df_ref_req['R3'].values[0]
 df_r3 = [r3, r3]

 r5 = df_ref_req['R5'].values[0]
 df_r5 = [r5, r5]

 r6 = df_ref_req['R6'].values[0]
 df_r6 = [r6, r6]

 fig2 = go.Figure()
 fig2.add_trace(go.Scatter(x=x_axis, y=nd,
                              mode='lines',
                              name='VI_1',
                line=dict(color='blue')))
 fig2.add_trace(go.Scatter(x=x_axis_vg, y=vg,
                              mode='lines',
                              name='VI_2',
                           line=dict(color='red')))
 fig2.add_trace(go.Scatter(x=df_ve, y=[-30, vgmax+5],
                          mode='lines',
                          showlegend=False,
                          line=dict(color='slategray', width=2, dash='dash')))
 fig2.add_trace(go.Scatter(x=df_v6, y=[-30, vgmax+5],
                          mode='lines',
                          showlegend=False,
                          line=dict(color='slategray', width=2, dash='dash')))
 fig2.add_trace(go.Scatter(x=df_v12, y=[-30, vgmax+5],
                          mode='lines',
                          showlegend=False,
                          line=dict(color='slategray', width=2, dash='dash')))
 fig2.add_trace(go.Scatter(x=df_vt, y=[-30, vgmax+5],
                          mode='lines',
                          showlegend=False,
                          line=dict(color='slategray', width=2, dash='dash')))
 fig2.add_trace(go.Scatter(x=df_r1, y=[-30, vgmax+5],
                          mode='lines',
                          showlegend=False,
                          line=dict(color='slategray', width=2, dash='dash')))
 fig2.add_trace(go.Scatter(x=df_r3, y=[-30, vgmax+5],
                          mode='lines',
                          showlegend=False,
                          line=dict(color='slategray', width=2, dash='dash')))
 fig2.add_trace(go.Scatter(x=df_r5, y=[-30, vgmax+5],
                          mode='lines',
                          showlegend=False,
                          line=dict(color='slategray', width=2, dash='dash')))
 fig2.add_trace(go.Scatter(x=df_r6, y=[-30, vgmax+5],
                          mode='lines',
                          showlegend=False,
                          line=dict(color='slategray', width=2, dash='dash')))

 fig2.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=[115,555,945,1350,1400,1925,2450,2700],
            ticktext=[115,555,945,1350,1400,1925,2450,2700],
            tickfont=dict(size=12),
            tickangle=65,),
        margin={'t': 25, 'l': 5, 'r': 5, 'b': 0},
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=1),
        paper_bgcolor="#E8F8F5",
        width=750,
        height=355,
        title_text="Corn Phenology",
        title_x=0.5,
        font={'color': "darkblue", 'family': "Arial"},
 )
 crp_plot.plotly_chart(fig2,use_column_width=True)

def temp_plot(key):
    STD_URL="C:/Users/Krupa/Desktop/riverbridge/Multifarm/combined/tbl_standard_deviation_weather_historical_data_26fields_29102020.csv"
    df_std=pd.read_csv(STD_URL,index_col=None)
    #df_std['hi_name']=pd.to_numeric(df_std['hi_name'])
    mask_std_key=(df_std['hi_name'] == key)
    df_std_key=df_std.loc[mask_std_key]
    df_std_key['hi_date']=pd.to_datetime(df_std_key['hi_date'],format='%d-%m-%Y')
    mask_std1=( df_std_key['hi_date'] >= begin_dt )
    mask_std2=( df_std_key['hi_date'] <= end_date)
    df_std1=df_std_key.loc[mask_std1]
    df_std2=df_std_key.loc[mask_std2]
    df_std_req=df_std1.append(df_std2)
    df_std_req.reset_index(drop=True,inplace=True)


    ac_temp_max=df_aw_cp['MaxTemp_C']
    ac_temp_min=df_aw_cp['MinTemp_C']
    avg_temp_max=df_avg_comb['hi_max_temp']
    avg_temp_max.reset_index(drop=True,inplace=True)
    avg_temp_min=df_avg_comb['hi_min_temp']
    avg_temp_min.reset_index(drop=True,inplace=True)
    std_max_1=df_std_req['hi_max_temp']
    std_max_2=df_std_req['hi_max_temp']*2
    std_min_1=df_std_req['hi_min_temp']
    std_min_2=df_std_req['hi_min_temp']*2



    df_tplot = pd.DataFrame(
        {'ac_temp_max': ac_temp_max,
         'ac_temp_min': ac_temp_min,
         'avg_temp_max': avg_temp_max,
         'avg_temp_min': avg_temp_min,
         'std_temp_max': std_max_1,
         'std_temp_min': std_min_1,
         'scnd_std_temp_max': std_min_1,
         'scnd_std_temp_min': std_min_2
        })


    df_tplot['diff_max']=abs(df_tplot['ac_temp_max'] - df_tplot['avg_temp_max'])
    df_tplot['diff_min']=abs(df_tplot['ac_temp_min'] - df_tplot['avg_temp_min'])


    conditions_max  = [ df_tplot['diff_max'] < df_tplot['std_temp_max'],
                        (df_tplot['diff_max'] >= df_tplot['std_temp_max']) & (df_tplot['diff_max'] <= df_tplot['scnd_std_temp_max']),
                         df_tplot['diff_max'] > df_tplot['scnd_std_temp_max']
                      ]
    values_max  = [ "green", 'gold', 'red' ]

    conditions_min  = [ df_tplot['diff_min'] < df_tplot['std_temp_min'],
                        (df_tplot['diff_min'] >= df_tplot['std_temp_min']) & (df_tplot['diff_min'] <= df_tplot['scnd_std_temp_min']),
                         df_tplot['diff_min'] > df_tplot['scnd_std_temp_min']
                      ]
    values_min  = [ "green", 'gold', 'red' ]

    df_tplot['color_max'] = np.select(conditions_max,values_max)
    df_tplot['color_min'] = np.select(conditions_min,values_min)
    len_temp=len(df_tplot)
    df_tplot['day_no']=np.arange(1,len_temp+1,1)

    x_axis = df_tplot['day_no']
    avg_temp_max = df_tplot['avg_temp_max']
    avg_temp_min = df_tplot['avg_temp_min']
    col_max = df_tplot['color_max']
    ac_temp_max = df_tplot['ac_temp_max']
    ac_temp_min = df_tplot['ac_temp_min']
    col_min = df_tplot['color_min']

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_axis, y=avg_temp_max,
                             mode='lines',
                             name='avg_temp_max',
                             line=dict(color='darkgray')))
    fig.add_trace(go.Scatter(x=x_axis, y=avg_temp_min,
                             mode='lines',
                             name='avg_temp_min',
                             line=dict(color='darkgray')))
    fig.add_trace(go.Scatter(x=x_axis, y=ac_temp_max,
                             mode='markers',
                             marker=dict(size=4, color=df_tplot['color_max'])))
    fig.add_trace(go.Scatter(x=x_axis, y=ac_temp_min,
                             mode='markers',
                             marker=dict(size=4, color= df_tplot['color_min'])))

    fig.update_layout(
        margin={'t': 26, 'l': 10, 'r': 10, 'b': 5},
        showlegend=False,
        paper_bgcolor="white",
        width=600,
        height=325,
        title_text="Daily Temperature Spread",
        title_x=0.5,
        font={'color': "darkblue", 'family': "Arial"},
    )
    t_plot.plotly_chart(fig,use_container_width=True)


def vi(fkey):
    df_vi = pd.read_csv(f"{mn_path}{vi_pth}{fkey}{vi_part}", index_col=False)
    df_vi.rename(columns={df_vi.columns[0]: "date", df_vi.columns[1]: "vi"}, inplace=True)
    df_vi['date'] = pd.to_datetime(df_vi['date'], format='%Y-%m-%d %H:%M:%S')
    xaxis = df_vi['date']
    vi = df_vi['vi']

    VI_DT = "C:/Users/Krupa/Desktop/riverbridge/Multifarm/VI_new/vi_dates.csv"
    df_vi_dt = pd.read_csv(VI_DT, index_col='crop_id', parse_dates=True)
    df_vi_dt['fm_harv_dt'] = pd.to_datetime(df_vi_dt['fm_harv_dt'], format='%d-%m-%Y')
    df_vi_dt['fm_sow_dt'] = pd.to_datetime(df_vi_dt['fm_sow_dt'], format='%d-%m-%Y')
    df_vi_dt['cal_sow_dt'] = pd.to_datetime(df_vi_dt['cal_sow_dt'], format='%d-%m-%Y')
    df_vi_dt['cal_harv_dt'] = pd.to_datetime(df_vi_dt['cal_harv_dt'], format='%d-%m-%Y')
    fm_sd = df_vi_dt.loc[fkey]['fm_sow_dt']
    mask_fms = (df_vi['date'] == fm_sd)
    fm_s1 = df_vi.loc[mask_fms]
    x1 = (fm_s1['date'])
    fm_s = (fm_s1['vi'])
    fm_sv = fm_s1['date']
    fm_sd_dt=datetime.strftime(fm_sd,format='%d-%m-%Y')

    fm_c = df_vi_dt.loc[fkey]['fm_harv_dt']
    mask_fmc = (df_vi['date'] == fm_c)
    fm_c1 = df_vi.loc[mask_fmc]
    fm_c_dt=datetime.strftime(fm_c,format='%d-%m-%Y')

    cal_s = df_vi_dt.loc[fkey]['cal_sow_dt']
    mask_cals = (df_vi['date'] == cal_s)
    cal_s1 = df_vi.loc[mask_cals]
    cal_s_dt=datetime.strftime(cal_s,format='%d-%m-%Y')

    cal_c = df_vi_dt.loc[fkey]['cal_harv_dt']
    mask_calc = (df_vi['date'] == cal_c)
    cal_c1 = df_vi.loc[mask_calc]
    cal_c_dt=datetime.strftime(cal_c,format='%d-%m-%Y')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xaxis, y=vi,
                             mode='lines',
                             name='ndvi'))
    fig.add_trace(go.Scatter(x=fm_s1['date'], y=fm_s1['vi'],
                             name=f"farmer_sow_dt: <b> {fm_sd_dt} </b>",
                             marker=dict(size=11, color='orangered')))
    fig.add_trace(go.Scatter(x=cal_s1['date'], y=cal_s1['vi'],
                             name=f"calc_sow_dt:  <b>{cal_s_dt}</b>",
                             marker=dict(size=7, color='yellow')))
    fig.add_trace(go.Scatter(x=fm_c1['date'], y=fm_c1['vi'],
                             name=f"farmer_harvest_dt: <b> {fm_c_dt} </b>",
                             marker=dict(size=11, color='firebrick')))
    fig.add_trace(go.Scatter(x=cal_c1['date'], y=cal_c1['vi'],
                             name=f"calc_harvest_dt: <b> {cal_c_dt} </b>",
                             marker=dict(size=7, color='darkorange')))

    fig.update_layout(
        margin={'t': 26, 'l': 5, 'r': 5, 'b': 10},
        legend=dict(yanchor="top", y=-0.1, xanchor="left", x=0),
        paper_bgcolor="white",
        width=570,
        height=535,
        font={'color': "darkslategray", 'family': "Arial"},
    )
    v_plot.plotly_chart(fig, use_container_width=True)

view_values =np.arange(45.0, 360.0, 1.5)
field_list=[2802,2803,2804,2806,2807,2808,2810,2811,2812,2813,2814,2817,2818,2819,2820,2823,2824,2827,2829,2830]
field_cmp=[2802,2803,2804,2806,2807,2808,2810,2811,2812,2813,2814,2815,2816,2817,2818,2819,2820,2823,2824,2825,2826,2827,2829,2830,'All']
TXT_URLS="C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/details_acerage.csv"
df_txt=pd.read_csv(TXT_URLS,index_col='crop_id')
VI_TXT="C:/Users/Krupa/Desktop/riverbridge/Multifarm/VI_new/vi_text.csv"
df_vi_txt=pd.read_csv(VI_TXT,index_col='crop_id')

G_URLS = {
    2802: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2802_google.png",
    2803: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2803_google.png",
    2804: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2804_google.png",
    2806: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2806_google.png",
    2807: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2807_google.png",
    2808: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2808_google.png",
    2810: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2810_google.png",
    2811: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2811_google.png",
    2812: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2812_google.png",
    2813: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2813_google.png",
    2814: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2814_google.png",
    2815: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2815_google.png",
    2816: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2816_google.png",
    2817: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2817_google.png",
    2818: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2818_google.png",
    2819: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2819_google.png",
    2820: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2820_google.png",
    2823: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2823_google.png",
    2824: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2824_google.png",
    2825: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2825_google.png",
    2826: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2826_google.png",
    2827: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2827_google.png",
    2829: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2829_google.png",
    2830: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2830_google.png"
}
SAT_URLS = {
    2802: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2802_sat.png",
    2803: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2803_sat.png",
    2804: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2804_sat.png",
    2806: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2806_sat.png",
    2807: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2807_sat.png",
    2808: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2808_sat.png",
    2810: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2810_sat.png",
    2811: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2811_sat.png",
    2812: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2812_sat.png",
    2813: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2813_sat.png",
    2814: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2814_sat.png",
    2815: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2815_sat.png",
    2816: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2816_sat.png",
    2817: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2817_sat.png",
    2818: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2818_sat.png",
    2819: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2819_sat.png",
    2820: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2820_sat.png",
    2823: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2823_sat.png",
    2824: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2824_sat.png",
    2825: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2825_sat.png",
    2826: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2826_sat.png",
    2827: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2827_sat.png",
    2829: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2829_sat.png",
    2830: "C:/Users/Krupa/Desktop/riverbridge/Multifarm/acerage/2830_sat.png"
}
mn_hd= """
    <h3 style="background-color: mediumseagreen;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b> Farm Dashboard </b></h3>"""

with st.beta_container():
   fkey = st.sidebar.selectbox("Current Farm", field_list, index=0)
   idx=field_cmp.index(fkey)
   ckey = st.sidebar.selectbox("Benchmark Farm", field_cmp, index=idx)
  # if(ckey=='All'):
    #   ckey=9090
   print(fkey,ckey)
   hd = df_txt.loc[fkey]['heading']
   st.markdown(hd, unsafe_allow_html=True)
   day_value = st.sidebar.empty()
   image_crn = st.sidebar.empty()
   start_date, end_date, begin_dt = sow_harvest_dates(fkey)
   df_loc, df_data, df_loc_sm, df_data_sm = get_data(fkey,ckey)
   df_aw, df_avg_farm, df_aw_cp, df_avg_comb = kpi_prep_data(fkey)
   mask_txt = df_txt.index == fkey
   df_txt_key = df_txt.loc[mask_txt]
   date = df_data.index[0]
   df_loc = df_loc.copy()
   df_loc_sm = df_loc_sm.copy()
   dt = df_txt.loc[fkey]['details']
   vi_dt = df_vi_txt.loc[fkey]['details']
   col1, col2, col3 = st.beta_columns([0.8, 0.8, 1])
   with col1:
      st.subheader("Farm Boundary")
      gimage = Image.open(G_URLS[fkey])
      gimage_alt=gimage.resize((490, 430))
      st.image(gimage_alt, caption="Google Earth Image: Farm AGB-2539",use_column_width=True)
      st.markdown(dt, unsafe_allow_html=True)

   with col2:
      st.subheader("Crop Boundary")
      simage = Image.open(SAT_URLS[fkey])
      simage_alt=simage.resize((490, 430))
      st.image(simage_alt, caption="Satellite Image: Farm AGB-2539",use_column_width=True)

   with col3:
       st.subheader("Vegetation Index")
       v_plot=st.empty()
       vi(fkey)
       #st.markdown(vi_dt, unsafe_allow_html=True)

with st.beta_container():
    st.markdown(heading2, unsafe_allow_html=True)
    gdd_plot = st.empty()
    t_plot = st.empty()
    gdd()
    temp_plot(fkey)

with st.beta_container():
    st.markdown(heading3, unsafe_allow_html=True)
    col1, col2 = st.beta_columns([1, 1])
    with col2:
        crp_plot = st.empty()
        crp(fkey)
        ph_plot = st.empty()
    with col1:
        deck_map = st.empty()
        k_plot = st.empty()
        animation_speed = None
        date_value = st.sidebar.empty()
        date_slider = st.sidebar.empty()
        if st.sidebar.checkbox("Timelapse"):
            animation_speed = 0.1
        i = itr.cycle(view_values)

        if animation_speed:
            for date in itr.cycle(df_data.index):
                time.sleep(animation_speed)
                render_slider(date)
                render_map(date, next(i))
                kpi_ind(date)
                plant(date)

        else:
            i = 45
            date = render_slider(date)
            render_map(date, i)
            kpi_ind(date)
            plant(date)
            ph_base(fkey, ckey, date)








