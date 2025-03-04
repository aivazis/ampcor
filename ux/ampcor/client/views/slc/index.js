// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2025 all rights reserved


// externals
import React, { useState } from 'react'
// locals
import styles from './styles'
import { useFlowContext } from '~/context'
import { SLC } from '~/panels'
import { NamedTools } from '~/widgets'


// explore the input SLCs
const slc = (props) => {
    // get the flow configuration
    const config = useFlowContext()
    // unpack the raster shapes
    const rasterShapes = { ref: config.reference.shape, sec: config.secondary.shape }
    // make an array of their labels
    const rasters = ["ref", "sec"]

    // select the slc to show
    const [slc, setSLC] = useState(rasters[0])
    // extract its shape
    const rasterShape = rasterShapes[slc]

    // specify the shape of individual tiles
    const tileShape = [512, 512]

    // build the tile uri stem
    const uri = `/slc/${slc}`

    // build the container, fill it, and return it
    return (
        <section style={styles.slc}>
            {/* the raster selector */}
            <NamedTools choices={rasters} select={setSLC} style={styles.sourceToolbox} />
            {/* the panel */}
            <SLC uri={uri}
                rasterShape={rasterShape} tileShape={tileShape}
                style={styles?.slcPanel} />
        </section>
    )
}


// publish
export default slc


// end of file
