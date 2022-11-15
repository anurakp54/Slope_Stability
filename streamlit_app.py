import pickle
from pathlib import Path
import streamlit_authenticator as stauth
import streamlit as st
from numpy import array
from pybimstab.slope import NaturalSlope
from pybimstab.watertable import WaterTable
from pybimstab.bim import BlocksInMatrix
from pybimstab.slipsurface import CircularSurface
from pybimstab.slipsurface import TortuousSurface
from pybimstab.slices import MaterialParameters, Slices
from pybimstab.slopestabl import SlopeStabl
import numpy as np
import pandas as pd


st.set_page_config(layout="wide")
#--- USER AUTHENTICATION --

names = ["CKST", "admin"]
usernames = ["CKST", "admin"]


file_path = Path(__file__).parent/"hashed_pw.pkl"

with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,"Scanner", "ttyyuu", cookie_expiry_days=30)

name, authentication_status, username = authenticator.login("login","sidebar")

if authentication_status == False:
    st.error("Username/Password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")

if authentication_status == True:
    authenticator.logout("Logout","sidebar")
    result = st.container()
    try:
        with st.sidebar:
            depth = st.number_input("Depth of Excavation", min_value=15, max_value=60, value=30, step=1)
            layer1 = st.number_input("Soil Layer Thickness", min_value=5, max_value=60, value=5, step=1)
            layer2 = st.number_input("Soft Rock Layer Thickness", min_value=5, max_value=60, value=5, step=1)
            calculate = st.button("Calculate")
            st.text('Licenses')
            st.image('data/Licenses.png')

        st.title("Slope Stability Analysis - without soil nail")
        st.subheader("Basic Design Parameters")
        st.image("data/C3.png")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Slope of each soil type")
            st.write("1. Slope in Soil (H:V) = 1.5:1")
            st.write("2. Slope in Rock (H:V) = 2:1")
            st.write("3. Slope in Hard Rock (H:V) = 4:1 ")



        with col2:
            st.subheader("Bench Width and Height")
            st.write("1. Bench in Soil at every 5 meters")
            st.write("2. Bench in Rock at every  5 meters")
            st.write("3. Bench in Hard Rock at every 5 meters")
            st.write(" Bench width will be 2.5 m., and every 15 m will be 10 m. bench")

            soil_data_df = pd.DataFrame(np.array(
                [['Residual soil', 26, 24, 20, 21], ['Unconsolidated Sediment', 40, 30, 21, 22],
                 ['Moderated Weather Silt Stone', 127, 32, 26, 27]]), columns=['type', 'cu', 'phi', 'gamma', 'gamma_block'])

            # """Define Soil Layer"""
            soil_layer1 = layer1
            soil_layer2 = layer2
            soil_layer3 = depth-layer1-layer2
            layer = np.array([soil_layer1,soil_layer2,soil_layer3])*(1/(soil_layer1+soil_layer2+soil_layer3))
            soil_data_df['cu'] = soil_data_df['cu'].astype(float)
            soil_data_df['phi'] = soil_data_df['phi'].astype(float)
            soil_data_df['gamma'] = soil_data_df['gamma'].astype(float)
            soil_data_df['gamma_block'] = soil_data_df['gamma_block'].astype(float)
            phi = (layer[0] * soil_data_df.loc[0]['phi'] + layer[1] * soil_data_df.loc[1]['phi'] + layer[2] *soil_data_df.loc[2]['phi'])
            cu = (layer[0] * soil_data_df.loc[0]['cu'] + layer[1] * soil_data_df.loc[1]['cu'] + layer[2] *soil_data_df.loc[2]['cu'])
            gamma = (layer[0] * soil_data_df.loc[0]['gamma'] + layer[1] * soil_data_df.loc[1]['gamma'] + layer[2] *soil_data_df.loc[2]['gamma'])
            gamma_block = (layer[0] * soil_data_df.loc[0]['gamma_block'] + layer[1] * soil_data_df.loc[1]['gamma_block'] +layer[2] * soil_data_df.loc[2]['gamma_block'])

            # """Define Slope"""
            slope1 = 1.5  # H:1
            slope2 = 1.0  # H:1
            slope3 = 0.25

            # """Typical Bench Width"""
            bench_height1 = 5
            bench_height2 = 5
            bench_height3 = 5

            """Geometry"""

            x = [0]
            y = [0]
            distance = 0

            lower = depth - soil_layer1 - soil_layer2
            lower_bench = int(lower / bench_height3)
            for bench in range(int(lower / bench_height3)):
                y.append((bench + 1) * bench_height3)
                x.append(slope3 * bench_height3)

            for bench in range(int(soil_layer2 / bench_height2)):
                y.append(y[-1] + bench_height2)
                x.append(slope2 * bench_height2)

            while y[-1] < depth:
                if y[-1] + bench_height1 <= depth:
                    y.append(y[-1] + bench_height1)
                    x.append(slope1 * bench_height1)
                else:
                    y.append(depth)
                    x.append((y[-1] - y[-2]) * slope1)

            y.reverse()
            x.reverse()
            # print(x)
            # print(y)
            x = x[:-1]
            x.insert(0, 0)
            for i in range(len(x) - 1):
                x[i + 1] = x[i] + x[i + 1]

            # print(x)
            # print(y)
            x.append(x[-1] + depth)
            x.insert(0, -depth)
            # print(f'Horizontal: {x}')

            y.append(0)
            y.insert(0, y[0])
            # print(f'Vertical: {y}')
            terrainCoordinates = array([x, y])
            slope = NaturalSlope(terrainCoordinates)
            bim = BlocksInMatrix(slopeCoords=slope.coords, blockProp=0.01, tileSize=0.5, seed=123)


            # print(slope.maxDepth())

            # Add bench
            def add_bench(x, y, height, bench_width):
                pos = y.index(height)
                y.insert(pos, y[pos])
                x.insert(pos, x[pos])
                posx = pos + 1
                for index in range(len(x[pos:]) - 1):
                    x[posx + index] = x[posx + index] + bench_width
                return x, y


            num_of_bench = len(y)
            bench_level = y[2:(num_of_bench - 2)]
            # print(bench_level)
            x1 = x
            y1 = y
            for lev in bench_level:
                x1, y1 = add_bench(x1, y1, lev, 2.5)

            # print(max(x1))
            # print(f'Horizontal:{x1}')
            # print(f'Vertical:v{y1}')
            terrainCoordinates = array([x1, y1])
            slope = NaturalSlope(terrainCoordinates)
            bim = BlocksInMatrix(slopeCoords=slope.coords, blockProp=0.05, tileSize=0.5, seed=123)

            # """Increasing Benching Width to 10 meters at every 15 m height"""
            # print(bench_level)
            bench_level.reverse()
            bench_at = bench_level[2::3]
            # print(bench_at)
            for bench in bench_at:
                x1, y1 = add_bench(x1, y1, bench, 7.5)

            # print(max(x1))
            # print(f'Horizontal:{x1}')
            # print(f'Vertical:v{y1}')
            terrainCoordinates = array([x1, y1])
            slope = NaturalSlope(terrainCoordinates)
            bim = BlocksInMatrix(slopeCoords=slope.coords, blockProp=0.05, tileSize=0.5, seed=123)
            # print(slope.maxDepth())

            A = -x1[0]
            B = x1[-1]
            # print(A,B)

            watertabDepths = array([[0, round(max(x1)) / 3, round(max(x1)) * 2 / 3, max(x1)],
                                    [2.5, 5, 5, 0]])
            watertable = WaterTable(slopeCoords=slope.coords,
                                    watertabDepths=watertabDepths,
                                    smoothFactor=2)
            # print(watertable.defineStructre())
            # print(f'furthest node: {max(x1)}')
            # print(f'max depth: {slope.maxDepth()}')
            preferredPath = CircularSurface(
                slopeCoords=slope.coords, dist1=A, dist2=B, radius=(B - A))
            surface = TortuousSurface(
                bim, dist1=A, dist2=B, heuristic='euclidean',
                reverseLeft=False, reverseUp=False, smoothFactor=1,
                preferredPath=preferredPath.coords, prefPathFact=1)
            material = MaterialParameters(
                cohesion=cu, frictAngle=phi, unitWeight=gamma,
                blocksUnitWeight=gamma_block, wtUnitWeight=9.8)
            slices = Slices(
                material=material, slipSurfCoords=surface.coords,
                slopeCoords=slope.coords, numSlices=20,
                watertabCoords=watertable.coords, bim=bim)
            stabAnalysis = SlopeStabl(slices, seedFS=2, Kh=0, maxLambda=1)
            Fsf, _lambdaf = stabAnalysis.getFm(stabAnalysis.FS['fs'], stabAnalysis.FS['lambda'])
            Fsm, _lambdam = stabAnalysis.getFf(stabAnalysis.FS['fs'], stabAnalysis.FS['lambda'])
            fig = stabAnalysis.plot()

            with result:
                fig
    except:
        with result:
            st.subheader("Try different parameters!")
            st.write('Also check if the thickness of three layers not exceed the depth!')
