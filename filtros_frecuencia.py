#formato 
#sample rate, bit depth, channels
#Archivo wav
#Leer wav en scipy
#Escribir wav en scipy
from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.io.wavfile import write

frecuencia_muestreo, muestras = wavfile.read("violin.wav")

print(frecuencia_muestreo)

print("Tipo: " + str(type(muestras)))
print("Dtype (bithdept): " + str(muestras.dtype))
print("shape = " + str(muestras.shape))
canales = 1
if len(muestras.shape) == 1:
    print("# Canales = 1")
else:
    print("# Canales = " + str(muestras.shape[1]))
    canales = muestras.shape[1]
duracion = muestras.shape[0] / frecuencia_muestreo
print("duracion:  " + "{:.2f}".format(duracion) + " segs")

tiempos = np.linspace(0., duracion, muestras.shape[0])


figura, ejes = plt.subplots(2,4)
if canales == 1:
    ejes[0,0].plot(tiempos, muestras, label="Canal mono")
else:
    ejes[0,0].plot(tiempos, muestras[:, 0], label="Izquierdo")
    ejes[0,0].plot(tiempos, muestras[:, 1], label="Derecho")

ejes[0,0].legend()
ejes[0,0].set(xlabel= "Tiempo (s)", ylabel="Amplitud")
#ejes[0,0].xlabel("Time [s]")
#ejes[0,0].ylabel("Amplitude")

if canales > 1:
    data = muestras[:,0]
else:
    data = muestras

cantidad_muestras = len(data)
periodo_muestreo = 1.0 / frecuencia_muestreo
transformada = np.fft.rfft(data)
frecuencias = np.fft.rfftfreq(cantidad_muestras, periodo_muestreo)

ejes[0,1].plot(frecuencias, np.abs(transformada), 
     label="Espectro original")
ejes[0,1].legend()
ejes[0,1].set(xlabel= "Frecuencia (Hz)", ylabel="Amplitud")

#Filtro para bajas 
#Hacer una copia
pasa_bajas = transformada.copy()
pasa_bajas[frecuencias > 2500] *= 0

pasa_bajas_data = np.fft.irfft(pasa_bajas)

ejes[1,1].plot(frecuencias, np.abs(pasa_bajas), 
     label="Espectro filtrado, pasa bajas")
ejes[1,1].legend()
ejes[1,1].set(xlabel= "Frecuencia (Hz)", ylabel="Amplitud")

ejes[1,0].plot(tiempos, pasa_bajas_data, label="Audio con pasa bajas")
ejes[1,0].legend()
ejes[1,0].set(xlabel="Tiempos (s)", ylabel="Amplitud")

write("pasa_bajas.wav", frecuencia_muestreo, pasa_bajas_data.astype(np.int16))

detiene_bandas = transformada.copy()
indices = ((frecuencias>1000) & (frecuencias<5000))
detiene_bandas[indices] *= 0
detiene_bandas_data = np.fft.irfft(detiene_bandas)
ejes[0,3].legend()
ejes[0,3].set(xlabel = "Frecuencia (hz)", ylabel = "Amplitud")

ejes[0,2].plot(tiempos, detiene_bandas_data, label="Audio con detiene bandas")
ejes[0,2].legend()
ejes[0,2].set(xlabel="Tiempos (s)", ylabel="Amplitud")

ejes[0,3].plot(frecuencias, np.abs(detiene_bandas), label = "Espectro detiene bandas")
ejes[0,3].legend()
ejes[0,3].set(xlabel="Frecuencia (hz)", ylabel="Amplitud")

write("detiene_bandas.wav", frecuencia_muestreo, detiene_bandas_data.astype(np.int16))

pasa_bajas_back = transformada.copy()
pasa_bajas_back[frecuencias < 2500] *= 0

pasa_bajas_back_data = np.fft.irfft(pasa_bajas_back)

ejes[1,2].plot(frecuencias, np.abs(pasa_bajas_back), 
     label="Espectro filtrado, pasa bajas al reves")
ejes[1,2].legend()
ejes[1,2].set(xlabel= "Frecuencia (Hz)", ylabel="Amplitud")

ejes[1,3].plot(tiempos, pasa_bajas_back_data, label="Audio con pasa bajas al reves")
ejes[1,3].legend()
ejes[1,3].set(xlabel="Tiempos (s)", ylabel="Amplitud")

write("pasa_bajas_back.wav", frecuencia_muestreo, pasa_bajas_back_data.astype(np.int16))

plt.show()