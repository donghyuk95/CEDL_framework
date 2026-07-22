
import os
import re
from glob import glob

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import joblib
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr
from matplotlib.ticker import FormatStrFormatter

from tensorflow import keras
from tensorflow.keras import layers, models

SSPLIST = [126, 245, 585]

home_dir = os.path.expanduser('~/')


def seq2dataset(seq, window_size):
    dataset = []
    for i in range(len(seq) - window_size + 1):
        subset = seq[i : (i + window_size)]
        dataset.append(subset)
    return np.array(dataset)


def domain_eastsea(da):
    try:
        lon_name = next(c for c in da.coords if "lon" in c.lower() or "xc" in c.lower())
        lat_name = next(c for c in da.coords if "lat" in c.lower() or "yc" in c.lower())
    except StopIteration:
        raise ValueError("error")

    print(f"Detected coordinates: {lon_name}, {lat_name}")

    lon = da[lon_name]
    lat = da[lat_name]

    da = da.where((lon >= 127) & (lon <= 143) & (lat >= 35) & (lat <= 52.1), drop=True)

    lon = da[lon_name]
    lat = da[lat_name]

    da = da.where((lat > 34.2) | (lon < 131.0), np.nan)
    da = da.where((lat > 35.0) | (lon < 132.0), np.nan)
    da = da.where((lat > 36.0) | (lon < 136.0), np.nan)
    da = da.where((lat > 42.7) | (lon < 140.3), np.nan)
    da = da.where((lat > 47.0) | (lon < 142.0), np.nan)
    da = da.where((lat > 50.0) | (lon < 142.5), np.nan)

    msk = da.where(da != 0, np.nan)
    return msk


def load_sst_fields(src_model, tgt_model, year_end):
    file_sst_src = home_dir + f"data/CMIP_npy/Dscaled_{src_model}_ssp*_SST_1995_{year_end}.nc"
    file_sst_tgt = home_dir + f"data/CMIP_npy/Dscaled_{tgt_model}_ssp*_SST_1995_{year_end}.nc"

    #! source data for using monthly climatology
    sst_src = xr.open_mfdataset(
        file_sst_src,
        combine="nested",
        concat_dim="ssp",
    )["THETA"]
    sst_src = sst_src.assign_coords(ssp=SSPLIST)
    sst_src = sst_src.rename({"XC": "lon", "YC": "lat"})
    sst_src = sst_src.where(sst_src.maskC)
    sst_src_mc = sst_src.groupby("time.month").mean("time")
    sst_src_histmc = sst_src.sel(time=slice("1995-04", "2014-12")).groupby('time.month').mean(["time"])
    sst_src_histavg = sst_src.sel(time=slice("1995-04", "2014-12")).mean(["time", "lat", "lon"])

    #! target data for validation
    sst_tgt = xr.open_mfdataset(
        file_sst_tgt,
        combine="nested",
        concat_dim="ssp",
    )["THETA"]
    sst_tgt = sst_tgt.assign_coords(ssp=SSPLIST)
    sst_tgt = sst_tgt.rename({"XC": "lon", "YC": "lat"})
    sst_tgt = sst_tgt.where(sst_tgt.maskC)
    sst_tgt_mc = sst_tgt.groupby("time.month").mean("time")
    sst_tgt_anom = sst_tgt.groupby("time.month") - sst_tgt_mc
    sst_tgt_anom_sm = sst_tgt_anom.mean(["lon", "lat"])

    return {
        "sst_src": sst_src,
        "sst_src_mc": sst_src_mc.compute(),
        "sst_src_histmc": sst_src_histmc.compute(),
        "sst_src_histavg": sst_src_histavg.compute(),
        "sst_tgt": sst_tgt,
        "sst_tgt_mc": sst_tgt_mc,
        "sst_tgt_anom": sst_tgt_anom,
        "sst_tgt_anom_sm": sst_tgt_anom_sm.compute(),
    }




def load_X_dataset(
    varlist=None,
    SSP=None,
    model="KACE-1-0-G",
    data_dir=None,
    set_input=20,
    useEOF85=0,
    year_start=1995,
):
    print("\n -----------Executing function: load_XandY----------")

    CEOF_TYPE = f"m15_ssp{SSP}"
    combined_df = None
    if useEOF85 == 0:
        fp1_suffix = f"PCscl_{year_start}_2100"
    elif useEOF85 == 1:
        fp1_suffix = f"PCscl_{year_start}_2080"

    for var in varlist:
        fp1 = os.path.expanduser(f"{data_dir}/commonEOF_results/CMIP_{CEOF_TYPE}_{var}_{fp1_suffix}.nc")

        with xr.open_dataset(fp1, decode_times=True) as ds:
            data = ds.sel(mode=slice(1, set_input), model=model)
            pcs = data["pcs"].values

        df = pd.DataFrame(
            pcs,
            index=data["time"],  # 원래 데이터의 time 좌표 유지
            columns=[f"{var}_PC{mm + 1}" for mm in range(set_input)],
        )
        df.index.name = "time"
        df.columns.name = "mode"
        combined_df = df if combined_df is None else pd.concat([combined_df, df], axis=1)
    return combined_df


def load_Y_dataset(
    SSP=None,
    data_dir=None,
    feature_st=0,
    feature_cnt=1,
    varlist_Y="SST",
    useEOF85=0,
    targetmodel="KACE-1-0-G",
    year_start=1995,
):
    print("\n -----------Executing function: load_Y_dataset----------")

    if useEOF85 == 0:
        filename = data_dir + f"/CMIP_EOF/EOF_Dscaled_rmSeason_{targetmodel}_ssp{SSP}_{varlist_Y}_{year_start}_2100.nc"
    elif useEOF85 == 1:
        filename = data_dir + f"/CMIP_EOF/EOF_Dscaled_rmSeason_{targetmodel}_ssp{SSP}_{varlist_Y}_{year_start}_2080.nc"

    output_en_tmp = feature_st + feature_cnt  # mode2 타겟시 1+1

    var = varlist_Y
    filename = os.path.expanduser(filename)

    with xr.open_dataset(filename, decode_times=True) as ds:
        data = ds.sel(mode=slice(1, output_en_tmp))
        pcs = data["pcs"].T  # (mode, time) -> (time, mode)
        eofs = data["eofs"]
        evrs = data["evrs"]
        wgts = data["wgts"]

    df_pcs = pd.DataFrame(
        pcs,
        index=data["time"],  # 원래 데이터의 time 좌표 유지
        columns=[f"{var}_PC{mm}" for mm in range(1, output_en_tmp + 1, 1)],
    )

    # 축 이름(Dimension Name) 지정
    df_pcs.index.name = "time"
    df_pcs.columns.name = "mode"

    return df_pcs, evrs, eofs, wgts

def load_eofs_pcs(
    src_model, tgt_model, SSPLIST, varlist_X, data_dir, year_start, nt, TIME, useEOF85=0,set_input=10
):
    """
    각 SSP 시나리오별로 X와 Y의 PCA/EOF 관련 데이터를 로드하고 병합하여 반환하는 함수
    """
    # 1. X 데이터는 딕셔너리로 유지
    pcs_X_src = {}
    pcs_X_tgt = {}

    # 2. Y 데이터는 결합(concat)을 위해 임시 리스트 사용
    _pcs_Y_src = []
    _evrs_Y_src = []
    _eofs_Y_src = []
    wgts_Y_src = None  # 마지막 루프의 가중치 값을 저장할 임시 변수

    # 3. 반복해서 쓰이는 인덱스와 좌표값 미리 정의
    ssp_idx = pd.Index(SSPLIST, name="ssp")
    mode_coords = list(range(1, nt + 1))

    for ssp in SSPLIST:
        #! Source 및 Target의 X 데이터 로드 (딕셔너리에 저장)
        pcs_X_src[ssp] = load_X_dataset(
            model=src_model,
            SSP=ssp,
            varlist=varlist_X,
            set_input=set_input,
            useEOF85=useEOF85,
            data_dir=data_dir,
            year_start=year_start,
        )
        pcs_X_tgt[ssp] = load_X_dataset(
            model=tgt_model,
            SSP=ssp,
            varlist=varlist_X,
            set_input=set_input,
            useEOF85=useEOF85,
            data_dir=data_dir,
            year_start=year_start,
        )

        #! Source의 Y 데이터 로드
        pcs, evrs, eofs, wgts_Y_src = load_Y_dataset(
            SSP=ssp,
            feature_st=0,  # 1 - 1
            feature_cnt=nt,
            targetmodel=src_model,
            varlist_Y="SST",
            useEOF85=useEOF85,
            data_dir=data_dir,
            year_start=year_start,
        )

        # DataArray로 변환 후 리스트에 추가
        pcs_da = xr.DataArray(
            pcs,
            dims=["time", "mode"],
            coords={"time": TIME, "mode": mode_coords},
        )

        _pcs_Y_src.append(pcs_da)
        _evrs_Y_src.append(evrs)
        _eofs_Y_src.append(eofs)
        wgts_Y_src = wgts_Y_src  # wgts는 SSP에 상관없이 동일할 것이므로 덮어씌움

    #! 리스트에 모인 Y 데이터들을 ssp 차원으로 최종 병합
    pcs_Y_src = xr.concat(_pcs_Y_src, dim=ssp_idx)
    eofs_Y_src = xr.concat(_eofs_Y_src, dim=ssp_idx)
    evrs_Y_src = xr.concat(_evrs_Y_src, dim=ssp_idx)

    return {
        "pcs_X_src": pcs_X_src,
        "pcs_X_tgt": pcs_X_tgt,
        "pcs_Y_src": pcs_Y_src.persist(),
        "eofs_Y_src": eofs_Y_src.persist(),
        "evrs_Y_src": evrs_Y_src.persist(),
        "wgts_Y_src": wgts_Y_src.persist(),
    }


def predict_target_pcs(
    SSPLIST,  ML_dir, pclen_loop, pcs_X_tgt, seq_length, var_trendX, TIME_seq, ssptransfer=None,src_model=None, tfmodel=None,useEOF85=None,set_input=10
):
    _pred_Y_tgt = []

    ssp_idx = pd.Index(SSPLIST, name="ssp")
    mode_coords = list(range(1, pclen_loop + 1))

    for ssp in SSPLIST:
        if ssptransfer is not None:
            ssp_sel = ssptransfer
        else:
            ssp_sel = ssp
        if useEOF85==1:
            name_useEOF85 = 'train2085_'
        else:
            name_useEOF85 = ''
            
        
        pred_list = []
        foldername = f"InPc{set_input}_Train_{src_model}_6xvar_SST_ssp{ssp_sel}_{name_useEOF85}{tfmodel}"
        print(foldername)
        print(f"{ML_dir}/{foldername}")
        foldername = glob(f"{ML_dir}/{foldername}")[0].split("/")[-1]
        folderdir = f"{ML_dir}/{foldername}/"

        mlp_path = folderdir + "PC1_mlp_scaler.joblib"
        file_mlp = joblib.load(mlp_path)
        trendmlp = file_mlp["trendmlp"]
        scaler_trend_X = file_mlp["scaler_trend_X"]
        scaler_trend_Y = file_mlp["scaler_trend_Y"]

        #! prediction by each PC mode
        for i in range(pclen_loop):
            print(f"SSP: {ssp} | Mode: {i}")  
            data = pcs_X_tgt[ssp]

            keras_path = folderdir + f"PC{i + 1}.keras"
            scaler_tf_path = folderdir + f"PC{i + 1}_tf_scaler.joblib"
            txt_path = folderdir + f"PC{i + 1}.csv"

            # * preprocessing 1. trend separation (only X_tgt)
            X_train, X_train_trend = preprocessing_seperate_trend(
                data, seq_length, var_trendX, feature_st=i
            )
            X_train_trend = scaler_trend_X.transform(X_train_trend)

            # * preprocessing 2. feature selection
            selectedvar_final = pd.read_csv(txt_path).columns
            X_train = X_train[selectedvar_final]

            # * preprocessing 3. scaling (only training)
            scaler_tf = joblib.load(scaler_tf_path)
            scalerX = scaler_tf["scalerX"]
            scalerY = scaler_tf["scalerY"]

            X_train = scalerX.transform(X_train)

            # * preprocessing 4. sequence (only training)
            X_train = seq2dataset(X_train, seq_length)

            print(X_train.shape)
            if tfmodel not in ['transformer','lstm']:
                X_train = X_train[:,-1,:] 
        
            # * prediction
            if tfmodel in ['xgboost','mlr','ridge']:
                xgb_path = folderdir + f"PC{i + 1}.joblib"  #
                trained_model = joblib.load(xgb_path)       #
                pred = trained_model.predict(X_train).reshape(-1, 1)  
            else:
                keras_path = folderdir + f"PC{i + 1}.keras"
                print(keras_path)
                trained_model = tf.keras.models.load_model(keras_path, safe_mode=False)
                pred = trained_model.predict(X_train)
            pred = scalerY.inverse_transform(pred).squeeze()

            if i == 0:  # * 2-1. prediction trend component
                X_train_trend = X_train_trend[: -seq_length + 1]
                pred_trend = trendmlp.predict(X_train_trend)
                pred_trend = scaler_trend_Y.inverse_transform(pred_trend)[:, 0]
                print(f"X_train_trend shape: {pred_trend.shape}")
                pred = pred + pred_trend

            pred_list.append(pred)

        pred_array = np.vstack(pred_list).T

        pred_da = xr.DataArray(
            pred_array,
            dims=["time", "mode"],
            coords={"time": TIME_seq, "mode": mode_coords},
        )
        _pred_Y_tgt.append(pred_da)

    pred_Y_tgt_final = xr.concat(_pred_Y_tgt, dim=ssp_idx)

    return pred_Y_tgt_final.compute()


def reconstructiont_eofs_cedl(pcs_src, eofs_src, weights, selmode1=1, selmode2=None, ssp_list=None):
    pcs_sliced = pcs_src.sel(mode=slice(selmode1, selmode2)).transpose("ssp", "time", "mode")
    eofs_sliced = eofs_src.sel(mode=slice(selmode1, selmode2)).transpose(
        "ssp", "mode", "lat", "lon"
    )

    pcs_vals = pcs_sliced.values
    eofs_vals = eofs_sliced.values

    # 2. 형상(Shape) 정보 획득
    nssp, ntime, nmode = pcs_vals.shape
    _, _, nlat, nlon = eofs_vals.shape

    # 3. 행렬 곱(Matrix Multiplication)을 통한 복원
    # (nssp, ntime, nmode) x (nssp, nmode, nlat*nlon) -> (nssp, ntime, nlat*nlon)
    recon_matmul = np.matmul(pcs_vals, eofs_vals.reshape(nssp, nmode, -1))
    recon_array = recon_matmul.reshape(nssp, ntime, nlat, nlon)

    # 4. ssp 리스트 지정 (지정되지 않은 경우 원본 coords에서 추출 시도)
    if ssp_list is None:
        ssp_list = pcs_sliced.coords.get("ssp", np.arange(nssp)).values

    # 5. Xarray DataArray로 변환
    recon_da = xr.DataArray(
        recon_array,
        dims=("ssp", "time", "lat", "lon"),
        coords={
            "ssp": ssp_list,
            "time": pcs_sliced.time,
            "lat": eofs_sliced.lat,
            "lon": eofs_sliced.lon,
        },
    )

    # 6. 시간축 슬라이싱 및 가중치 제거 (Weight Un-weighting)
    recon_da = recon_da / weights

    return recon_da





def load_src_true_eof_recons(tgt_model, selmodes, sst_tgt, sst_src_mc):
    """
    선택된 EOF 모드(selmodes)를 이용해 SSP별 데이터를 한 번에 로드하고,
    SST를 재구성(Anomaly + Seasonality)하여 반환하는 함수
    """
    # 1. SSPLIST 순서에 맞춰 파일 경로 리스트 생성
    file_paths = [
        f"{home_dir}data/CMIP_EOF/EOF_Dscaled_rmSeason_{tgt_model}_ssp{ssp}_SST_1995_2100.nc"
        for ssp in SSPLIST
    ]

    # 2. 여러 파일을 ssp 차원으로 한 번에 병합하여 로드
    with xr.open_mfdataset(file_paths, combine="nested", concat_dim="ssp") as da:
        da = da.assign_coords(ssp=SSPLIST)

        eofs_tmp = da.patterns.sel(mode=slice(1, selmodes))
        pcs_tmp = da.pcs.sel(mode=slice(1, selmodes))
        wgts_tmp = da.wgts

        # 3. 재구성 연산 (xr.dot을 쓰면 ssp 루프 없이 전체 차원에 대해 한 번에 행렬곱 수행)
        recon_tmp = (xr.dot(pcs_tmp, eofs_tmp, dims="mode") / wgts_tmp).where(sst_tgt.maskC)

        # Anomaly 저장 (한 번의 compute()로 모든 SSP 연산을 동시 처리)
        recon_DD_8modes_tgt_rm = recon_tmp.compute()

    # 4. 계절성 더하기 (xarray가 알아서 ssp 차원끼리 매칭하여 계산함)
    recon_DD_8modes_tgt = recon_DD_8modes_tgt_rm.groupby("time.month") + sst_src_mc

    return {
        "recon_DD_8modes_tgt_rm": recon_DD_8modes_tgt_rm,
        "recon_DD_8modes_tgt": recon_DD_8modes_tgt,
    }



def load_pcs_x_ensemble(
    src_model, tgt_model, SSPLIST, varlist_X, data_dir, year_start, nt, TIME, useEOF85=0
):
    """
    각 SSP 시나리오별로 X와 Y의 PCA/EOF 관련 데이터를 로드하고 병합하여 반환하는 함수
    """
    # 1. X 데이터는 딕셔너리로 유지
    pcs_X_src = {}
    pcs_X_tgt = {}


    for ssp in SSPLIST:
        pcs_X_tgt[ssp] = load_X_dataset(
            model=tgt_model,
            SSP=ssp,
            varlist=varlist_X,
            set_input=10,
            useEOF85=useEOF85,
            data_dir=data_dir,
            year_start=year_start,
        )



    return {
        "pcs_X_src": pcs_X_src,
        "pcs_X_tgt": pcs_X_tgt,
        "pcs_Y_src": pcs_Y_src.persist(),
        "eofs_Y_src": eofs_Y_src.persist(),
        "evrs_Y_src": evrs_Y_src.persist(),
        "wgts_Y_src": wgts_Y_src.persist(),
    }

def load_src_true_eof_recons(tgt_model,selmodes, sst_mc_tgt):
    # 1. SSPLIST 순서에 맞춰 파일 경로 리스트 생성
    file_paths = [
        f"{home_dir}data/CMIP_EOF/EOF_Dscaled_rmSeason_{tgt_model}_ssp{ssp}_SST_1995_2100.nc"
        for ssp in SSPLIST
    ]
    # file_paths = sorted(file_paths, 
    print(file_paths)

    # 2. 여러 파일을 ssp 차원으로 한 번에 병합하여 로드
    with xr.open_mfdataset(file_paths, combine="nested", concat_dim="ssp") as da:
        da = da.assign_coords(ssp=SSPLIST)

        eofs_tmp = da.eofs
        pcs_tmp = da.pcs
        wgts_tmp = da.wgts
        evrs_tmp = da.evrs

        # 3. 재구성 연산 (xr.dot을 쓰면 ssp 루프 없이 전체 차원에 대해 한 번에 행렬곱 수행)
        # recon_tmp = (xr.dot(pcs_tmp, eofs_tmp, dims="mode") / wgts_tmp).where(sst_tgt.maskC)
    print()
    recon_tmp = reconstructiont_eofs_cedl(
        pcs_tmp, eofs_tmp, wgts_tmp, selmode1=1, selmode2=selmodes, ssp_list=SSPLIST
    )
    # Anomaly 저장 (한 번의 compute()로 모든 SSP 연산을 동시 처리)
    recon_DD_8m_rm = recon_tmp.compute()
    recon_DD_8m = recon_DD_8m_rm.groupby("time.month") + sst_mc_tgt

    return {"recon_DD_8m_rm": recon_DD_8m_rm, "recon_DD_8m": recon_DD_8m
            , 'eofs': eofs_tmp, 'pcs': pcs_tmp, 'wgts': wgts_tmp}
    
    
    
    
    
    

@keras.utils.register_keras_serializable()
class AbsolutePositionalEncoding(layers.Layer):  #! 원래 되던것
    def __init__(self, max_len=5000, d_model=None, **args):
        super().__init__(**args)
        self.max_len = max_len
        self.d_model = d_model
        self.pos_encoding = None

    def build(self, input_shape):
        seq_len, self.d_model = input_shape[-2], input_shape[-1]
        self.max_len = max(self.max_len, seq_len)

        # NumPy로 포지셔널 인코딩 계산 (더 안정적인 방법)
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * (-np.log(10000.0) / self.d_model))

        pos_encoding = np.zeros((seq_len, self.d_model))

        # 짝수 인덱스에 사인 적용
        pos_encoding[:, 0::2] = np.sin(position * div_term)

        # 홀수 인덱스에 코사인 적용 (d_model이 홀수인 경우 고려)
        cos_indices = np.arange(1, self.d_model, 2)
        pos_encoding[:, 1::2] = np.cos(position * div_term[: len(cos_indices)])

        self.pos_encoding = self.add_weight(
            shape=(seq_len, self.d_model),
            initializer=keras.initializers.Constant(pos_encoding),
            trainable=False,
            name="positional_encoding",
        )

    def call(self, inputs):
        return inputs + self.pos_encoding

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len, "d_model": self.d_model})
        return config
    
    

@keras.utils.register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, head_size, num_heads, ff_dim, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

    def build(self, input_shape):
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)

        # 커스텀 멀티헤드 어텐션 초기화
        self.mha = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.head_size, dropout=self.dropout)

        # 피드포워드 네트워크
        self.ffn = keras.Sequential(
            [
                layers.Dense(self.ff_dim, activation="gelu"),
                layers.Dense(input_shape[-1]),
            ]
        )
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout_layer = layers.Dropout(self.dropout)

    def call(self, inputs, rel_pos_embeddings=None):
        # 레이어 정규화
        norm_input = self.ln1(inputs)

        # 어텐션 계산 (상대 위치 임베딩 통합)
        attn_output = self.mha(
            query=norm_input,
            value=norm_input,
            key=norm_input,
            attention_mask=rel_pos_embeddings,  # 상대 위치 임베딩 주입
        )

        # 잔차 연결 및 정규화
        attn_output = self.dropout_layer(attn_output)
        res1 = inputs + attn_output
        norm_res = self.ln2(res1)

        # 피드포워드 네트워크
        ffn_output = self.ffn(norm_res)
        ffn_output = self.dropout_layer(ffn_output)
        return res1 + ffn_output

def create_model(dlmodel=None, input_shape=None, **modelconfig):
    head_size = modelconfig.get("head_size")
    num_heads = modelconfig.get("num_heads")
    num_transformer_blocks = modelconfig.get("num_transformer_blocks")
    ff_dim = modelconfig.get("ff_dim")
    mlp_units = modelconfig.get("mlp_units")
    dropout = modelconfig.get("dropout")
    learning_rate = modelconfig.get("learning_rate")
    feature_final = modelconfig.get("feature_final")
    LOSS_fn = modelconfig.get("LOSS_fn")

    print("\n -----------Executing function: create_model----------")

    if dlmodel == "transformer":
        print('transformer')
        inputs = layers.Input(shape=input_shape)
        x = inputs
        x = AbsolutePositionalEncoding()(x)

        for i in range(num_transformer_blocks):
            x = TransformerBlock(
                head_size=head_size,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                name=f"transformer_block_{i}",
            )(x, None)

        x = layers.Lambda(lambda x: x[:, -1, :])(x)

        for dim in mlp_units:
            x = layers.Dense(dim, activation="gelu")(x)
            x = layers.Dropout(dropout)(x)

        outputs = layers.Dense(feature_final, name="output_layer")(x)
        model = models.Model(inputs=inputs, outputs=outputs)



    if dlmodel in ["transformer", "lstm", "mlp"]:
        model.compile(
            loss=LOSS_fn,
            optimizer=keras.optimizers.Adam(learning_rate),
        )
    else:
        pass

    return model


def preprocessing_seperate_trend(  #! only training purpose
    df,
    seq_length=4,
    var_list="",
    feature_st=0,
):
    print("\n -----------Executing function: preprocessing_trend----------")

    # 원본 인덱스와 형태를 그대로 보존하면서 복사본 생성
    df_res = df.copy()
    df_trend = pd.DataFrame(index=df.index)  # 인덱스 일치시키기

    # 트렌드 분리 루프 (여러 컬럼이든 단일 컬럼이든 동일하게 작동)
    for var in var_list:
        if var not in df.columns:
            continue  # 안전장치: 존재하지 않는 컬럼 패스

        tmp_detre, tmp_trend = polyfit_3rd(df[var])

        # squeeze() 후 원본 데이터프레임의 구조에 맞게 대입
        df_res[var] = tmp_detre.squeeze()
        df_trend[var] = tmp_trend.squeeze()

    return df_res, df_trend


def polyfit_3rd(y, Npoly=3):
    nt = len(y)
    x = np.arange(1, nt + 1, 1)
    coeffs = np.polyfit(x, y, Npoly)  # 3차 다항식 계수 계산
    poly = np.poly1d(coeffs)  # 다항식 객체 생성
    y_pred = poly(x)  # x 길이에 맞는 예측값 생성

    residuals = y - y_pred  # 잔차 계산

    return residuals, y_pred
