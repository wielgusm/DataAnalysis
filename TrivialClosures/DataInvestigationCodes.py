import numpy as np
import pandas as pd
import sys
sys.path.append('/home/maciek/eat')
from eat.hops import util as hu
from eat.io import hops, util
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.signal as scs

def ImportBaselineData120(nameF):
    #Loads baseline data from file
    VisibilityData = hu.pop120(nameF) 
    ParamData = hu.params(nameF)
    print([ParamData.sbd, ParamData.mbd, ParamData.delay])

    trot = ParamData.trot
    frot = ParamData.frot[120]
    
    return VisibilityData, trot, frot

def ImportBaselineData212(nameF):
    #Loads baseline data from file
    VisibilityData = hu.pop212(nameF) 
    ParamData = hu.params(nameF)

    trot = ParamData.trot
    frot = ParamData.frot[212]
    
    return VisibilityData, trot, frot 

def ImportCorrectTri120(nameF1,nameF2,nameF3,TrimV = -26):
    Vis1_0, trot1, frot1  = ImportBaselineData120(nameF1)
    Vis2_0, trot2, frot2  = ImportBaselineData120(nameF2)
    Vis3_0, trot3, frot3  = ImportBaselineData120(nameF3)

    Vis1 = CorrectPhases120(Vis1_0,trot1,frot1)
    Vis2 = CorrectPhases120(Vis2_0,trot2,frot2)
    Vis3 = CorrectPhases120(Vis3_0,trot3,frot3)
    #if TrimV==0:
    Vis1 = Vis1[TrimV:,:,:]
    Vis2 = Vis2[TrimV:,:,:]
    Vis3 = Vis3[TrimV:,:,:]  
    
    return Vis1, Vis2, Vis3
    
    

def CorrectPhases120(VisibilityData,trot,frot):

    PhaseCorrect = VisibilityData[:,:,:]*(frot[:,np.newaxis,:])*(trot[np.newaxis,:,np.newaxis])

    return PhaseCorrect

def RotateBack212(VisibilityData,trot):

    #PhaseCorrect = VisibilityData[:,:]*np.conj(trot[:,np.newaxis])
    PhaseCorrect = VisibilityData[:,:]*(trot[:,np.newaxis])
    
    return PhaseCorrect
'''
def AvArr(NArr,step,Nax):

    
    #print('dudu',[Nax, np.shape(NArr)])
    NdirAv = NArr.shape[Nax]/step #How many buckets

    #print([NArr.shape[Nax],step,NdirAv])
    ANArr = np.empty(np.shape(np.take(NArr,range(1), axis=Nax)))

    for cou in range(0,NdirAv): 
        MeanV = np.take(NArr,range(cou,cou+step), axis=Nax)   
        MeanV = np.mean(MeanV,axis=Nax,keepdims=True)
        ANArr = np.concatenate((ANArr,MeanV), axis=Nax)
        #print('cou=', cou)

    ANArr = np.delete(ANArr,range(1),axis = Nax)
    return ANArr
'''
def AvArr(NArr,step,Nax):
    NdirAv = int(np.floor(NArr.shape[Nax]/step)) #How many buckets
    ANArr = np.empty(np.shape(np.take(NArr,range(1), axis=Nax)))
    for cou in range(NdirAv): 
        MeanV = np.take(NArr,range(step*cou,step*cou+step), axis=Nax)   
        MeanV = np.mean(MeanV,axis=Nax,keepdims=True)
        ANArr = np.concatenate((ANArr,MeanV), axis=Nax)
    ANArr = np.delete(ANArr,range(1),axis = Nax)
    return ANArr

def AvArrVar(NArr,step,Nax):
    NdirAv = int(np.floor(NArr.shape[Nax]/step)) #How many buckets
    ANArr = np.empty(np.shape(np.take(NArr,range(1), axis=Nax)))
    STDArr = np.empty(np.shape(np.take(NArr,range(1), axis=Nax)))
    for cou in range(NdirAv): 
        LocalV = np.take(NArr,range(step*cou,step*cou+step), axis=Nax)   
        MeanV = np.mean(LocalV,axis=Nax,keepdims=True)
        VarV = np.var(LocalV,axis=Nax,keepdims=True)
        ANArr = np.concatenate((ANArr,MeanV), axis=Nax)
        STDArr = np.concatenate((STDArr,VarV), axis=Nax)
    ANArr = np.delete(ANArr,range(1),axis = Nax)
    STDArr = np.sqrt(np.delete(STDArr,range(1),axis = Nax))/np.sqrt(2.*step)
    #this is sigma of the result mean, one axis sigma
    return ANArr, STDArr


def AvArr2(NArr,step1,Nax1, step2, Nax2):
    AvArr1 = AvArr(NArr,step1,Nax1)
    AvArr2 = AvArr(AvArr1,step2,Nax2)
    return AvArr2

def AvArr2Band(NArr,stepT, NumChan):
    AvArr1 = AvArr(NArr,stepT,1)
    AvArr2 = AvArr(AvArr1,128,2)
    AvArr3 = AvArr(AvArr2,NumChan,0)
    return AvArr3
    

def Bispectrum(VisA,VisB,VisC,step1=1,Nax1=1,step2=1,Nax2=2):

    AvVisA = AvArr2(VisA,step1,Nax1,step2,Nax2)
    AvVisB = AvArr2(VisB,step1,Nax1,step2,Nax2)
    AvVisC = AvArr2(VisC,step1,Nax1,step2,Nax2)
    Bisp = AvVisA*AvVisB*AvVisC
    return Bisp

def BispectrumBand(VisA,VisB,VisC,step1=1,Nax1=1,NChan=1):

    AvVisA = AvArr2Band(VisA,step1,NChan)
    AvVisB = AvArr2Band(VisB,step1,NChan)
    AvVisC = AvArr2Band(VisC,step1,NChan)
    Bisp = AvVisA*AvVisB*AvVisC
    return Bisp

def SDreductionVis212(VisA, vecT, vecF):

    MeanVisAav2 = np.zeros((len(vecT),len(vecF)))
    stdV = np.zeros((len(vecT),len(vecF)))
    nV = np.zeros((len(vecT),len(vecF)))
    
    for couT in range(len(vecT)):
        for couF in range(len(vecF)):
            VisAav = AvArr2(VisA,vecT[couT],0,vecF[couF],1)
            #Bisp = Bispectrum(VisA,VisB,VisC, vecT[couT],1,vecF[couF],2)
            VisAav2 = np.abs(VisAav)**2
            MeanVisAav2[couT,couF] = np.sqrt(np.mean(VisAav2))
            #MeanVisAav[couT,couF] = np.abs(np.mean(VisAav))
            #meanV[couT,couF] = np.mean(np.mean(VisAav.flatten()))
            #stdV[couT,couF] = np.std(np.angle(Bisp.flatten()))
            nV[couT,couF] = vecT[couT]*vecF[couF]
    return MeanVisAav2, nV


def SDreduction(VisA, VisB, VisC, vecT, vecF):

    meanCP = np.zeros((len(vecT),len(vecF)))
    stdCP = np.zeros((len(vecT),len(vecF)))
    nCP = np.zeros((len(vecT),len(vecF)))
    
    for couT in range(len(vecT)):
        for couF in range(len(vecF)):
            Bisp = Bispectrum(VisA,VisB,VisC, vecT[couT],1,vecF[couF],2)
            meanCP[couT,couF] = np.mean(np.angle(Bisp.flatten()))
            stdCP[couT,couF] = np.std(np.angle(Bisp.flatten()))
            nCP[couT,couF] = vecT[couT]*vecF[couF]
    return meanCP, stdCP, nCP
    
def SDreductionBand(VisA, VisB, VisC, vecT):

    meanCP = np.zeros((len(vecT),len(vecB)))
    stdCP = np.zeros((len(vecT),len(vecB)))
    nCP = np.zeros((len(vecT),len(vecB)))
    
    for couT in range(len(vecT)):
        for couB in range(len(vecB)):
            Bisp = BispectrumBand(VisA,VisB,VisC, vecT[couT],1,vecB[couB])
            meanCP[couT,couB] = np.mean(np.angle(Bisp.flatten()))
            stdCP[couT,couB] = np.std(np.angle(Bisp.flatten()))
            nCP[couT,couB] = 128*vecT[couT]*vecB[couB]
    return meanCP, stdCP, nCP

def ChangesAveraging(Vis, vecT, vecB, what='SNR'):

    #meanCP = np.zeros((len(vecT),len(vecB)))
    SNRV = np.zeros((len(vecT),len(vecB)))
    DBAV = np.zeros((len(vecT),len(vecB)))
    SSVDV = np.zeros((len(vecT),len(vecB)))
    nV = np.zeros((len(vecT),len(vecB)))
    BAV = np.zeros((len(vecT),len(vecB)))

    for couT in range(len(vecT)):
        for couB in range(len(vecB)):
            VisAv = AvArr2(Vis,vecT[couT],0,vecB[couB],1)
            SNRV[couT,couB] = SNR(VisAv.flatten())
            DBAV[couT,couB] = DebAmp(VisAv.flatten())
            #SSVDV[couT,couB] = np.sqrt(2)*SSTD(VisAv.flatten())/np.std(VisAv.flatten())
            SSVDV[couT,couB] = SSTDD(VisAv.flatten())
            #SSVDV[couT,couB] = np.std(VisAv.flatten())
            #stdCP[couT,couB] = np.std(np.angle(Bisp.flatten()))
            nV[couT,couB] = vecT[couT]*vecB[couB]
            #BAV[couT,couB] = np.sqrt(np.mean(np.abs(VisAv.flatten())**2))
            BAV[couT,couB] = np.mean(np.abs(VisAv.flatten()))
    return SNRV, DBAV, SSVDV, nV,BAV

    
def PhaseCorr(phase,MaxDel = 100):

    dph = np.zeros(MaxDel)
    for couD in range(MaxDel):
        phase1 = phase[:len(phase)-couD]
        phase2 = phase[couD:]
        dph[couD] = np.mean((phase1 - phase2)**2)

    return dph, np.asarray(range(MaxDel))
        
        
    


def PrintInfoSample(BispA):

    print('Test of zero mean: ', stats.ttest_1samp(np.angle(BispA.flatten()), 0.))
    #stats.ttest_1samp(np.random.normal(0.0, 1.0, 1000), 0.)
    print('Mean = ', np.mean(np.angle(BispA.flatten()))) 
    print('sd = ', np.std(np.angle(BispA.flatten()))) 
    print('Mean in deg = ', (180./np.pi)*np.mean(np.angle(BispA.flatten()))) 
    print('sd in deg = ',  (180./np.pi)*np.std(np.angle(BispA.flatten()))) 
    print('Number of samples =', np.size(BispA))
    #print('Estimated full sd in deg =', (180./np.pi)*np.std(np.angle(BispA.flatten()))/np.sqrt(np.size(BispA)))
    

def SNR(vect):
    #vect = vect.flatten()
    #SNR = np.sqrt(np.mean(vect.real)**2 + np.mean(vect.imag)**2)/np.sqrt(np.std(vect.real)**2 + np.std(vect.imag)**2)
    SNR=DebAmp(vect)/SSTDD(vect)
    return SNR

def MomentsReal(vect):
    import scipy.stats as stat
    mv = np.mean(vect.real)
    sv = np.std(vect.real)
    zv = stat.skew(vect.real)*sv**3
    kv = stat.kurtosis(vect.real)*sv**4
    return mv,sv,zv,kv

def SSTD(sample,axis=0):
    #SAMPLE STD
    #D_sample = np.diff(sample,axis=0)
    #SSTD = np.std(np.hstack(((D_sample.real).flatten(),(D_sample.imag).flatten())))/np.sqrt(2)
    SSTD = np.std(sample)/np.sqrt(2)
    return SSTD

def SSTDD(sample,axis=0):
    #SAMPLE STD
    D_sample = np.diff(sample,axis=0)
    SSTDD = np.std(np.hstack(((D_sample.real).flatten(),(D_sample.imag).flatten())))/np.sqrt(2)

    return SSTDD

def DebAmp(X):
    DbA = np.sqrt(np.mean(np.abs(X.flatten())**2) - 2.*SSTDD(X)**2)
    #DbA = np.sqrt(np.mean(np.abs(X.flatten())**2) - np.std(X)**2)
    return DbA


def loadBaseline(baseline, path, polar='RR',typ='212'):
    try:
        pathA = path[:-12]+'alist.5s'
        alist = hops.read_alist(pathA)
        print(pathA)
    except (IOError):
        pathA = path[:-12]+'alist.v6'
        alist = hops.read_alist(pathA)
    
    
    scan_id = path[-7:-1]
    idx = (alist["scan_id"]==scan_id)&(alist["polarization"]==polar)&(alist["baseline"]==baseline)
    aa = np.array(alist.loc[idx, ["extent_no", "root_id"]])
    #print(aa)
    FileName = baseline+'.B.'+str(aa[0,0])+'.'+aa[0,1]
    FilePath = path+FileName
    #print(FilePath)
    snr = hu.params(FilePath).snr
    if typ=='212':
        Vis = hu.pop212(FilePath)
    else:
        Vis = hu.pop120(FilePath)
    return Vis, snr


def ReadTriangleFolder(ObsPath,nameFA,ZJSswitch=True,TrimV=-27):
    
    a = hops.read_alist(nameFA); " ".join(a.columns); idx = a["scan_id"]==ObsPath; scan_idd = ObsPath;
    
    NPnames = np.array(a.loc[idx, ["baseline", "extent_no", "root_id"]])
    
    filenames = []
    for x in range(NPnames.shape[0]):
        filenames.append(str(NPnames[x,0])+'.B.'+str(NPnames[x,1])+'.'+str(NPnames[x,2])) 
    
    if ZJSswitch == True:
        ZJbase = [s for s in filenames if "ZJ" in s]; ZJname = str(ZJbase[0])
        JSbase = [s for s in filenames if "JS" in s]; JSname = str(JSbase[0])
        ZSbase = [s for s in filenames if "ZS" in s]; ZSname = str(ZSbase[0])        
    else:
        JLbase = [s for s in filenames if "JL" in s]; JLname = str(JLbase[0])
        LSbase = [s for s in filenames if "LS" in s]; LSname = str(LSbase[0])
        JSbase = [s for s in filenames if "JS" in s]; JSname = str(JSbase[0])
        
        

        #TRIANGLES
        
    if ZJSswitch == True:
        ###ZJS###
        TriName = 'ZJS'
        BaselineA1 = "/mnt/ssd/links/apr2016c/3554/"+ObsPath+"/"+ZJname
        BaselineA2 = "/mnt/ssd/links/apr2016c/3554/"+ObsPath+"/"+JSname
        BaselineA3 = "/mnt/ssd/links/apr2016c/3554/"+ObsPath+"/"+ZSname#conjugate
        dfChan1 = 2.*np.pi*hu.params(BaselineA1).delay*(hu.params(BaselineA1).fedge - hu.params(BaselineA1).ref_freq - 32.)
        dfChan2 = 2.*np.pi*hu.params(BaselineA2).delay*(hu.params(BaselineA2).fedge - hu.params(BaselineA2).ref_freq - 32.)
        dfChan3 = 2.*np.pi*hu.params(BaselineA3).delay*(hu.params(BaselineA3).fedge - hu.params(BaselineA3).ref_freq - 32.)
        #hu.params(BaselineA3).nchan
        Trim_df = np.amin(np.array([len(dfChan1),len(dfChan2),len(dfChan3)]))
        dfChan =  dfChan1[-Trim_df:] + dfChan2[-Trim_df:] - dfChan3[-Trim_df:]
    else:
        ###LJS###
        TriName = 'LJS'
        BaselineA1 = "/mnt/ssd/links/apr2016c/3554/"+ObsPath+"/"+JLname#conjugate
        BaselineA2 = "/mnt/ssd/links/apr2016c/3554/"+ObsPath+"/"+JSname
        BaselineA3 = "/mnt/ssd/links/apr2016c/3554/"+ObsPath+"/"+LSname#conjugate
        dfChan1 = 2.*np.pi*hu.params(BaselineA1).delay*(hu.params(BaselineA1).fedge - hu.params(BaselineA1).ref_freq - 32.)
        dfChan2 = 2.*np.pi*hu.params(BaselineA2).delay*(hu.params(BaselineA2).fedge - hu.params(BaselineA2).ref_freq - 32.)
        dfChan3 = 2.*np.pi*hu.params(BaselineA3).delay*(hu.params(BaselineA3).fedge - hu.params(BaselineA3).ref_freq - 32.)

        Trim_df = np.amin(np.array([len(dfChan1),len(dfChan2),len(dfChan3)]))
        dfChan = - dfChan1[-Trim_df:] + dfChan2[-Trim_df:] - dfChan3[-Trim_df:]
        #dtnc = -hu.params(BaselineA1).delay + hu.params(BaselineA2).delay -hu.params(BaselineA3).delay
        #########

    Vis1, Vis2, Vis3 = ImportCorrectTri120(BaselineA1,BaselineA2,BaselineA3,TrimV)


    return Vis1, Vis2, Vis3, dfChan


def PlotCP0_diagnosticsLJS(Vis1,Vis2,Vis3,dfChan,BandShift = 2,Tav=5,ObsPath='096-0500'):
    import matplotlib.pyplot as plt
    sh = 0; sh23 = BandShift;  #sh25 = 2; Tav = 5
    Bsp1ch = BispectrumBand(np.conj(Vis1[-15-11+sh:-15+12+sh,:,:]),Vis2[-15-11+sh:-15+12+sh,:,:],np.conj(Vis3[-15-11+sh:-15+12+sh,:,:]),Tav,1,1)
    Bsp1chs = BispectrumBand(np.conj(Vis1[-15-11+sh23:-15+12+sh23,:,:]),Vis2[-15-11+sh23:-15+12+sh23,:,:],np.conj(Vis3[-15-11+sh23:-15+12+sh23,:,:]),Tav,1,1)
    Bsp25ch = BispectrumBand(np.conj(Vis1[-15-11:-15+12,:,:]),Vis2[-15-11:-15+12,:,:],np.conj(Vis3[-15-11:-15+12,:,:]),Tav,1,23)
    Bsp25chs = BispectrumBand(np.conj(Vis1[-15-11+sh23:-15+12+sh23,:,:]),Vis2[-15-11+sh23:-15+12+sh23,:,:],np.conj(Vis3[-15-11+sh23:-15+12+sh23,:,:]),Tav,1,23)
    Bsp25ch = Bsp25ch.squeeze();Bsp1ch = Bsp1ch.squeeze(); Bsp1chs = Bsp1chs.squeeze()
    #print([np.shape(Bsp1ch),np.shape(Bsp25ch)])
    Bsp1chc = Bsp1ch*np.exp(1j*dfChan[-15-11+sh:-15+12+sh,np.newaxis])
    Bsp1chsc = Bsp1chs*np.exp(1j*dfChan[-15-11+sh23:-15+12+sh23,np.newaxis])
    #print(np.shape(Bsp1chc))
    Bsp23 = AvArr(Bsp1chsc,23,0)
    #print(np.shape(Bsp23))
    Bsp25chsc = Bsp25chs*np.exp(1j*dfChan[-15-11+sh23:-15+12+sh23])
    #Bsp25chc = Bsp25ch*np.exp(1j*dfChan[-15-11:-15+12])
    BspB = np.mean((Bsp1ch),1)
    BspBc = np.mean((Bsp1chc),1)

    plt.subplot(121)
    plt.plot(np.angle(BspB),'*')
    plt.axhline(y=0., xmin=0, xmax=1200, linewidth=2, color = 'k')
    plt.xlabel('Freq channel')
    plt.ylabel('Mean CP')
    plt.title('Before correction')
    plt.subplot(122)
    #plt.plot(np.angle(Bsp1chc.flatten(order='C')),'*')
    plt.plot(np.angle(BspBc),'*')
    plt.axhline(y=0., xmin=0, xmax=1200, linewidth=2, color = 'k')
    plt.title('After delay correction')
    
    plt.suptitle('LJS '+'('+ObsPath+')',fontsize=24,y = 1.08)
    plt.show()

    #plt.subplot(121)
    plt.plot(np.real(Bsp1chc.flatten()),np.imag(Bsp1chc.flatten()), '*',label ='5s, avg 1 channel')
    plt.axhline(y=0., xmin=0, xmax=60, linewidth=2, color = 'k')
    plt.axvline(x=0., ymin=-40, ymax=40, linewidth=2, color = 'k')
    #plt.axis('equal')
    #plt.figure(figsize=(5,5))
    #plt.show()
    #plt.subplot(122)
    plt.plot(np.real(Bsp25chs.flatten()),np.imag(Bsp25chs.flatten()),'r*',label='5s, avg all channels, shifted band')
    plt.plot(np.mean(np.real(Bsp1ch.flatten())),np.mean(np.imag(Bsp1ch.flatten())),'ks',markersize = 10,label='mean avg 1ch')
    plt.plot(np.mean(np.real(Bsp25ch.flatten())),np.mean(np.imag(Bsp25ch.flatten())),'go',markersize = 10,label='mean avg all ch')
    plt.plot(np.mean(np.real(Bsp25chs.flatten())),np.mean(np.imag(Bsp25chs.flatten())), 'c^',markersize = 10,label='mean avg all ch, shifted band')
    #plt.axhline(y=0., xmin=0, xmax=60, linewidth=2, color = 'k')
    #plt.axvline(x=0., ymin=-40, ymax=40, linewidth=2, color = 'k')
    
    #plt.figure(figsize=(10,10))
    fig = plt.gcf()
    fig.set_size_inches(8, 7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.axis('equal')
    plt.title('Measured bispectra on complex plane')
    plt.show()
    print('Average over whole band shifted by 2 channels, no correction of non closing delay:')
    print('Mean CP [deg]:', (180/np.pi)*np.angle(np.mean(Bsp25chs.flatten())) )
    print('Error of the mean CP [deg]:', (180/np.pi)*np.std(np.angle(Bsp25chs.flatten()))/np.sqrt(len(Bsp25chs.flatten())))
    print('\n')
    print('Average over whole band shifted by 2 channels, correction of the non closing delay:')
    print('Mean CP [deg]:', (180/np.pi)*np.angle(np.mean(Bsp23.flatten())) )
    print('Error of the mean CP [deg]:', (180/np.pi)*np.std(np.angle(Bsp23.flatten()))/np.sqrt(len(Bsp23.flatten())))

    print('\n')
    print('Average over symmetric band, no correction of non closing delay:')
    print('Mean CP [deg]:', (180/np.pi)*np.angle(np.mean(Bsp25ch.flatten())) )
    print('Error of the mean CP [deg]:', (180/np.pi)*np.std(np.angle(Bsp25ch.flatten()))/np.sqrt(len(Bsp25ch.flatten())))
    
    print('\n')
    print('Average over single channels in symmetric band, no correction of the non closing delay:')
    print('Mean CP [deg]:', (180/np.pi)*np.angle(np.mean(Bsp1ch.flatten())) )
    print('Error of the mean CP [deg]:', (180/np.pi)*np.std(np.angle(Bsp1ch.flatten()))/np.sqrt(len(Bsp1ch.flatten())))

    print('\n')
    print('Average over single channels in symmetric band, correction of the non closing delay:')
    print 'Mean CP [deg]:', (180/np.pi)*np.angle(np.mean(Bsp1chc.flatten())) 
    print('Error of the mean CP [deg]:', (180/np.pi)*np.std(np.angle(Bsp1chc.flatten()))/np.sqrt(len(Bsp1chc.flatten())))

    
    #print([(180/np.pi)*np.angle(np.mean(Bsp1ch.flatten())), (180/np.pi)*np.std(np.angle(Bsp1ch.flatten()))/np.sqrt(len(Bsp1ch.flatten()))])
    #print([(180/np.pi)*np.angle(np.mean(Bsp1chc.flatten())), (180/np.pi)*np.std(np.angle(Bsp1chc.flatten()))/np.sqrt(len(Bsp1chc.flatten()))])
    #print([(180/np.pi)*np.angle(np.mean(Bsp25ch.flatten())), (180/np.pi)*np.std(np.angle(Bsp25ch.flatten()))/np.sqrt(len(Bsp25ch.flatten()))])
    #print([(180/np.pi)*np.angle(np.mean(Bsp25chsc.flatten())), (180/np.pi)*np.std(np.angle(Bsp25chsc.flatten()))/np.sqrt(len(Bsp25chsc.flatten()))])


    
def PlotCP0_diagnosticsZJS(Vis1,Vis2,Vis3,dfChan,BandShift = 2,Tav=5,ObsPath='096-0500'):
    import matplotlib.pyplot as plt
    sh = 0; sh23 = BandShift;  #sh25 = 2; Tav = 5
    Bsp1ch = BispectrumBand((Vis1[-15-11+sh:-15+12+sh,:,:]),Vis2[-15-11+sh:-15+12+sh,:,:],np.conj(Vis3[-15-11+sh:-15+12+sh,:,:]),Tav,1,1)
    Bsp1chs = BispectrumBand((Vis1[-15-11+sh23:-15+12+sh23,:,:]),Vis2[-15-11+sh23:-15+12+sh23,:,:],np.conj(Vis3[-15-11+sh23:-15+12+sh23,:,:]),Tav,1,1)
    Bsp25ch = BispectrumBand((Vis1[-15-11:-15+12,:,:]),Vis2[-15-11:-15+12,:,:],np.conj(Vis3[-15-11:-15+12,:,:]),Tav,1,23)
    Bsp25chs = BispectrumBand((Vis1[-15-11+sh23:-15+12+sh23,:,:]),Vis2[-15-11+sh23:-15+12+sh23,:,:],np.conj(Vis3[-15-11+sh23:-15+12+sh23,:,:]),Tav,1,23)
    Bsp25ch = Bsp25ch.squeeze();Bsp1ch = Bsp1ch.squeeze(); Bsp1chs = Bsp1chs.squeeze()
    #print([np.shape(Bsp1ch),np.shape(Bsp25ch)])
    Bsp1chc = Bsp1ch*np.exp(1j*dfChan[-15-11+sh:-15+12+sh,np.newaxis])
    Bsp1chsc = Bsp1chs*np.exp(1j*dfChan[-15-11+sh23:-15+12+sh23,np.newaxis])
    #print(np.shape(Bsp1chc))
    Bsp23 = AvArr(Bsp1chsc,23,0)
    #print(np.shape(Bsp23))
    Bsp25chsc = Bsp25chs*np.exp(1j*dfChan[-15-11+sh23:-15+12+sh23])
    #Bsp25chc = Bsp25ch*np.exp(1j*dfChan[-15-11:-15+12])
    BspB = np.mean((Bsp1ch),1)
    BspBc = np.mean((Bsp1chc),1)

    plt.subplot(121)
    plt.plot(np.angle(BspB),'*')
    plt.axhline(y=0., xmin=0, xmax=1200, linewidth=2, color = 'k')
    plt.xlabel('Freq channel')
    plt.ylabel('Mean CP')
    plt.title('Before correction')
    plt.subplot(122)
    #plt.plot(np.angle(Bsp1chc.flatten(order='C')),'*')
    plt.plot(np.angle(BspBc),'*')
    plt.axhline(y=0., xmin=0, xmax=1200, linewidth=2, color = 'k')
    plt.title('After delay correction')
    plt.suptitle('ZJS '+'('+ObsPath+')',fontsize=24,y = 1.08)
    plt.show()

    #plt.subplot(121)
    plt.plot(np.real(Bsp1chc.flatten()),np.imag(Bsp1chc.flatten()), '*',label ='5s, avg 1 channel')
    plt.axhline(y=0., xmin=0, xmax=60, linewidth=2, color = 'k')
    plt.axvline(x=0., ymin=-40, ymax=40, linewidth=2, color = 'k')
    #plt.axis('equal')
    #plt.figure(figsize=(5,5))
    #plt.show()
    #plt.subplot(122)
    plt.plot(np.real(Bsp25chs.flatten()),np.imag(Bsp25chs.flatten()),'r*',label='5s, avg all channels, shifted band')
    plt.plot(np.mean(np.real(Bsp1ch.flatten())),np.mean(np.imag(Bsp1ch.flatten())),'ks',markersize = 10,label='mean avg 1ch')
    plt.plot(np.mean(np.real(Bsp25ch.flatten())),np.mean(np.imag(Bsp25ch.flatten())),'go',markersize = 10,label='mean avg all ch')
    plt.plot(np.mean(np.real(Bsp25chs.flatten())),np.mean(np.imag(Bsp25chs.flatten())),'c^',markersize = 10,label='mean avg all ch, shifted band')
    #plt.axhline(y=0., xmin=0, xmax=60, linewidth=2, color = 'k')
    #plt.axvline(x=0., ymin=-40, ymax=40, linewidth=2, color = 'k')
    
    #plt.figure(figsize=(10,10))
    fig = plt.gcf()
    fig.set_size_inches(8, 7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.axis('equal')
    plt.title('Measured bispectra on complex plane')
    plt.show()
    print('Average over whole band shifted by 2 channels, no correction of non closing delay:')
    print('Mean CP [deg]:', (180/np.pi)*np.angle(np.mean(Bsp25chs.flatten())) )
    print('Error of the mean CP [deg]:', (180/np.pi)*np.std(np.angle(Bsp25chs.flatten()))/np.sqrt(len(Bsp25chs.flatten())))
    print('\n')
    print('Average over whole band shifted by 2 channels, correction of the non closing delay:')
    print('Mean CP [deg]:', (180/np.pi)*np.angle(np.mean(Bsp23.flatten())) )
    print('Error of the mean CP [deg]:', (180/np.pi)*np.std(np.angle(Bsp23.flatten()))/np.sqrt(len(Bsp23.flatten())))
    print('\n')
    
    print('Average over symmetric band, no correction of non closing delay:')
    print('Mean CP [deg]:', (180/np.pi)*np.angle(np.mean(Bsp25ch.flatten())) )
    print('Error of the mean CP [deg]:', (180/np.pi)*np.std(np.angle(Bsp25ch.flatten()))/np.sqrt(len(Bsp25ch.flatten())))

    print('\n')
    print('Average over single channels in symmetric band, no correction of the non closing delay:')
    print('Mean CP [deg]:', (180/np.pi)*np.angle(np.mean(Bsp1ch.flatten())) )
    print('Error of the mean CP [deg]:', (180/np.pi)*np.std(np.angle(Bsp1ch.flatten()))/np.sqrt(len(Bsp1ch.flatten())))

    print('\n')
    print('Average over single channels in symmetric band, correction of the non closing delay:')
    print 'Mean CP [deg]:', (180/np.pi)*np.angle(np.mean(Bsp1chc.flatten())) 
    print('Error of the mean CP [deg]:', (180/np.pi)*np.std(np.angle(Bsp1chc.flatten()))/np.sqrt(len(Bsp1chc.flatten())))



def GetCoherenceTime(Vis212):
    
    s = SSTDD(Vis212)/np.sqrt(Vis212.shape[1])
    V = AvArr(Vis212,Vis212.shape[1],1)
    vecT = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30,35,40,50,60,80,100]; vecF = [V.shape[1]]; 
    SNRV, DBAV, STDV, nV, BAV = ChangesAveraging(V,vecT,vecF)
    DBAV = np.sqrt(BAV.flatten()**2 - 2.*s**2/(vecT))
    
    #coherence time from non-debiased amplitude
    test = BAV-0.9*BAV[0]
    tcoh = np.where(test[:-1] * test[1:] <= 0)
    try:
        indT = int(tcoh[0][0])
        tcoh = float(vecT[indT]) + float(vecT[indT+1]-vecT[indT])*test[indT]/(-test[indT+1]+test[indT]) -1.
        tcoh1 = tcoh[0]
    except IndexError:
        tcoh1 = np.nan
    if all(test>0):
        tcoh1=100.
    
    #coherence time from debiased amplitude
    test2 = DBAV-0.9*np.amax(DBAV)
    tcoh2 = np.where((test2[:-1] * test2[1:] <= 0)& (test2[:-1] > 0))
    try:
        indT2 = int(tcoh2[0][0])
        tcoh2 = float(vecT[indT2]) + float(vecT[indT2+1]-vecT[indT2])*test2[indT2]/(-test2[indT2+1]+test2[indT2]) -1.
    except IndexError:
        tcoh2 = np.nan
    if all(test2>0):
        tcoh2=100.
        
    return tcoh1, tcoh2


def PlotCoherenceTimescale(V212, logscaley = False):
    s = SSTDD(V212)/np.sqrt(V212.shape[1])
    V = AvArr(V212,V212.shape[1],1)
    vecT = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,25,30,35,40,45,50,60,70,80,100]; vecF = [V.shape[1]]; 
    SNRV, DBAV, STDV, nV, BAV = ChangesAveraging(V,vecT,vecF)
    DBAV = np.sqrt(BAV.flatten()**2 - 2.*s**2/(vecT))
    
    #plot amplitudes vs averaging time
    plt.figure(figsize=(14,6))
    plt.errorbar(nV/V.shape[1], DBAV, xerr=0.0, yerr=1*s*np.sqrt(2)/np.sqrt(vecT),fmt='-bo',label='debiased')
    plt.errorbar(nV/V.shape[1], BAV, xerr=0.0, yerr=1*s*np.sqrt(2)/np.sqrt(vecT),fmt='-ro',label='non-debiased')
    plt.xscale('log'); plt.grid();
    if Logscaley==True:
        plt.yscale('log')
    plt.ylabel('Debiased amp.',fontsize=15)
    plt.xlabel('coherent intgr. time $t_{av}$ [s]',fontsize=15)
    plt.axhline(y=0.9*np.amax(DBAV),color='b',linestyle='--', label='90 percent of max debiased')
    plt.axhline(y=0.9*BAV[0],color='r',linestyle='--', label='90 percent of 1s biased')
    plt.legend()
    plt.show()
    
    #print info
    test = BAV-0.9*BAV[0]
    tcoh = np.where(test[:-1] * test[1:] <= 0)
    try:
        indT = int(tcoh[0][0])
        tcoh = float(vecT[indT]) + float(vecT[indT+1]-vecT[indT])*test[indT]/(-test[indT+1]+test[indT]) -1.
        tcoh1 = tcoh[0]
    except IndexError:
        tcoh1 = np.nan
    if all(test>0):
        tcoh1=100.
    
    test2 = DBAV-0.9*np.amax(DBAV)
    tcoh2 = np.where((test2[:-1] * test2[1:] <= 0)& (test2[:-1] > 0))
    try:
        indT2 = int(tcoh2[0][0])
        tcoh2 = float(vecT[indT2]) + float(vecT[indT2+1]-vecT[indT2])*test2[indT2]/(-test2[indT2+1]+test2[indT2]) -1.
    except IndexError:
        tcoh2 = np.nan
    if all(test2>0):
        tcoh2=100.
    print 'Charasteristic std for coherent averaging over 1s, all channels: ', s
    print 'Coherence time from non-debiased amplitudes [s]: ', tcoh1
    print 'Coherence time from debiased amplitudes [s]: ', tcoh2

    #autocorrelation plots   
    V = V212
    c = scs.correlate2d(V,np.conj(V),mode='same')
    fig, ax = plt.subplots(1,2,figsize=(10, 10))
    cax0 = ax[0].imshow(np.abs(c), interpolation='nearest',aspect = 0.07*420/V212.shape[0])
    ax[0].set_xlabel('frequency')
    ax[0].set_ylabel('time')
    ax[0].set_title('sqrt autocorrelation')
    autocorT = np.sum(np.abs(c),1)
    autocorTn = autocorT/np.amax(autocorT)
    t = np.linspace(1, len(autocorT),len(autocorT)) - (len(autocorT)/2.)

    ax[1].plot(t,np.sqrt(autocorTn),'-*')
    ax[1].axis([-15,15,0.7,1.03])
    ax[1].set_aspect(85)
    ax[1].axhline(y=0.9,color='k')
    ax[1].axvline(x=0.0,color='k')
    ax[1].set_xlabel('time [s]')
    ax[1].set_title('sqrt autocorrelation')
    plt.show() 

def CorrectDelayLin(V):
    #correct delay
    foo = AvArr(V,V.shape[1],1) #assume that axis 1 is frequency channels
    unPh = np.unwrap((np.angle(foo.flatten())))
    t = np.linspace(1,len(unPh),len(unPh))
    a,b = np.polyfit(t,unPh,1)
    Vc = V*np.exp(-1j*(a*t+b))[:,np.newaxis]
    return Vc

def get_list_file_paths(alist,path0='',conditions="(alist['baseline']!= 'Small off duty Czechoslovakian traffic warden')"):
    #e.g., listP = get_list_file_paths(alist,path0, "(alist['baseline']=='AX')")
    exec('cond='+conditions)
    expt_no = list(alist.loc[cond,'expt_no'])
    scan_id = list(alist.loc[cond,'scan_id'])
    baseline = list(alist.loc[cond,'baseline'])
    extent_no = list(alist.loc[cond,'extent_no'])
    root_id = list(alist.loc[cond,'root_id'])  
    path_list = [path0+'/'+str(expt_no[x])+'/'+str(scan_id[x])+'/'+baseline[x]+'.B.'+str(extent_no[x])+'.'+str(root_id[x]) for x in range(len(expt_no))]
    return path_list