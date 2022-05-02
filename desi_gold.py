import os
#import runtime
import numpy as np
import astropy.units as u

from   astropy.coordinates import SkyCoord
from   astropy.table       import Table, vstack, hstack, unique, join
from   ros_tools           import tile2rosette, calc_rosr
from   gama_limits         import gama_field
from   cartesian           import cartesian, rotate
from   cosmo               import cosmo, distmod
from   lss                 import fetch_lss
from   survey              import survey_specifics

def desi_gold():
    from   desiutil.dust                 import mwdust_transmission
    from   desitarget.sv3.sv3_targetmask import desi_mask, bgs_mask
    #from   desitarget.targetmask          import desi_mask, bgs_mask

    root   = os.environ['DESI_ROOT'] + '/spectro/redux/everest/healpix/'
    tpix   = Table.read(root + 'tilepix.fits')

    tiles  = np.arange(1000)
    ros    = np.array([tile2rosette(x) for x in tiles])
    
    # https://desi.lbl.gov/trac/wiki/SurveyOps/OnePercent
    # G12: [1,2]; G15: [8,9,10, 17]
    
    uros   = np.unique(ros)
    uros   = uros[uros>-1]
    
    gama   = np.isin(ros, uros)

    tiles  = tiles[gama]
    
    tpix   = tpix[np.isin(tpix['TILEID'].data, tiles)]
    hps    = np.unique(tpix['HEALPIX'].data)

    root  += '/sv3/bright/'

    fpaths = [root + '{}/{}/redrock-sv3-bright-{}.fits'.format(str(x)[:3], x, x) for x in hps]
    fpaths = [x for x in fpaths if os.path.exists(x)]

    # e.g. 280/28027/redrock-sv3-bright-28027.fits
    tabs   = []

    print('Gathering DESI zs.')

    for x in fpaths:
        zbest = Table.read(x, hdu='REDSHIFTS')
        fmap  = Table.read(x, hdu='FIBERMAP')
        efmap = Table.read(x, hdu='EXP_FIBERMAP')

        tids  = np.unique(zbest['TARGETID'])

        # row ordered.                                                                                                                                                                              
        assert  np.all(zbest['TARGETID'] == fmap['TARGETID'])
    
        efmap['ROS'] = tile2rosette(efmap['TILEID'].data)
    
        efmap_tid = efmap['TARGETID', 'ROS'] 
        efmap_tid = unique(efmap_tid, keys=['TARGETID'])
        efmap_tid.sort('TARGETID')

        assert  np.all(efmap_tid['TARGETID'].data == tids)
    
        del  fmap['TARGETID']
    
        zbest = hstack([zbest, fmap])
    
        zbest = join(zbest, efmap_tid, join_type='left', keys='TARGETID')
        zbest['ROS_DIST'] = 1.e4 * np.ones_like(zbest['Z'])
    
        '''
        for rosn in np.unique(efmap['ROS'].data):
            # TODO: Small rosette overlap.                                                                                                                                                         
            isin  = zbest['ROS'].data == rosn
            zbest['ROS_DIST'][isin] = calc_rosr(rosn, zbest['TARGET_RA'].data[isin], zbest['TARGET_DEC'].data[isin])
        '''
    
        tabs.append(zbest)
    
    desi_zs = vstack(tabs)

    # remove skies.                                                                                                                                                                                 
    print('Sky frac: 1-{:.6f}'.format(np.mean(desi_zs['TARGETID'].data >= 0)))

    desi_zs = desi_zs[desi_zs['TARGETID'].data >= 0]    
    
    # TODO: Problem here
    #desi_zs = desi_zs[(desi_zs['MAIN_BGS_TARGET'].data & bgs_mask['BGS_BRIGHT']) != 0]

    desi_zs['GOOD_Z'] = (desi_zs['ZWARN'].data == 0) & (desi_zs['DELTACHI2'].data > 40)

    desi_zs['ZDESI'] = desi_zs['Z']

    del  desi_zs['Z']

    desi_zs['FIELD'] = [f'R{xx}' for xx in desi_zs['ROS'].data]

    ##  Cut DESI to good redshifts.                                                                                                                                                                
    desi_zs['GAMA_FIELD'] = gama_field(desi_zs['TARGET_RA'].data, desi_zs['TARGET_DEC'].data)

    ##  HACK: PHOTSYS ASSUMED S.                                                                                                                                                                     
    desi_zs['GMAG_DRED']  = 22.5 - 2.5 * np.log10(desi_zs['FLUX_G'].data / mwdust_transmission(desi_zs['EBV'].data, 'G', 'S', match_legacy_surveys=True))
    desi_zs['RMAG_DRED']  = 22.5 - 2.5 * np.log10(desi_zs['FLUX_R'].data / mwdust_transmission(desi_zs['EBV'].data, 'R', 'S', match_legacy_surveys=True))

    desi_zs['W1MAG_DRED']  = 22.5 - 2.5 * np.log10(desi_zs['FLUX_W1'].data / mwdust_transmission(desi_zs['EBV'].data, 'W1', 'S', match_legacy_surveys=True))
    desi_zs['W2MAG_DRED']  = 22.5 - 2.5 * np.log10(desi_zs['FLUX_W2'].data / mwdust_transmission(desi_zs['EBV'].data, 'W2', 'S', match_legacy_surveys=True))
    
    desi_zs['GMR']        = desi_zs['GMAG_DRED'] - desi_zs['RMAG_DRED']
    desi_zs['DETMAG']     = desi_zs['RMAG_DRED']
    
    desi_zs['LEGACYPET']  = desi_zs['RMAG_DRED'] + survey_specifics('desi')['pet_offset']
    
    desi_zs['IN_GOLD']    = desi_zs['GOOD_Z'].data & (desi_zs['ZDESI'] > 0.039)  & (desi_zs['ZDESI'] < 0.263)

    clustering, full      = fetch_lss(pprint=False, sort=False) 

    clustering_ids        = np.unique(clustering['TARGETID'].data)
    full_ids              = np.unique(full['TARGETID'].data)

    desi_zs['IN_CLUSTERING'] = np.isin(desi_zs['TARGETID'].data, clustering_ids)
    desi_zs['IN_FULL']       = np.isin(desi_zs['TARGETID'].data, full_ids)

    clustering_cols = ['WEIGHT_ZFAIL', 'WEIGHT', 'NZ']
    full_cols       = ['NTILE', 'TILES', 'TILELOCIDS', 'LOCATION_ASSIGNED', 'TILELOCID_ASSIGNED', 'COMP_TILE', 'FRACZ_TILELOCID', 'BITWEIGHTS', 'PROB_OBS']
                       
    for cols, cat in zip([clustering_cols, full_cols], [clustering, full]):
        cols       += ['TARGETID']                       
        desi_zs     = join(desi_zs, cat[cols], keys='TARGETID')
                       
    ##  Archive step. 
    archive = Table(desi_zs, copy=True)

    xyz     = cartesian(desi_zs['TARGET_RA'].data, desi_zs['TARGET_DEC'].data, desi_zs['ZDESI'].data)

    desi_zs['CARTESIAN_X'] = xyz[:,0]
    desi_zs['CARTESIAN_Y'] = xyz[:,1]
    desi_zs['CARTESIAN_Z'] = xyz[:,2]

    xyz     = rotate(desi_zs['TARGET_RA'].data, desi_zs['TARGET_DEC'].data, xyz)

    desi_zs['ROTCARTESIAN_X'] = xyz[:,0]
    desi_zs['ROTCARTESIAN_Y'] = xyz[:,1]
    desi_zs['ROTCARTESIAN_Z'] = xyz[:,2]

    desi_zs['LUMDIST'] = cosmo.luminosity_distance(desi_zs['ZDESI'].data)
    desi_zs['DISTMOD'] = distmod(desi_zs['ZDESI'].data)
    
    desi_zs.pprint()

    # TODO: FIND_FILE
    root  = os.environ['CSCRATCH'] + '/norberg/'
    fpath = root + '/GAMA4/gama_gold.fits'
    opath = fpath.replace('gama_gold', 'desi_main_gold')

    print('Writing {}'.format(opath))

    desi_zs.write(opath, format='fits', overwrite=True)


if __name__ == '__main__':
    desi_gold()

    print('Done.')


