// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// externals
import React, { useState } from 'react'
// locals
import styles from './styles'
import { useFlowContext } from '~/context'
import SLC from '~/widgets/slc'
import Zoom from '~/widgets/zoom'


// explore the input SLCs
const slc = (props) => {
    // get the flow configuration
    const config = useFlowContext()
    // unpack the raster shapes
    const rasterShapes = { ref: config.reference.shape, sec: config.secondary.shape }

    // select the slc to show
    const [slc, setSLC] = useState("ref")
    // extract its shape
    const rasterShape = rasterShapes[slc]

    // our tile shape; optimized so that each line is one page
    const tileShape = [512, 512]

    // the zoom level suggestions for each axis
    const maxZoom  = Math.min(
        ...rasterShape.map((sz, i) => Math.floor(Math.log2(sz/tileShape[i])))
    )

    // start out at maximum zoom
    const [zoom, setZoom] = useState(maxZoom)
    // convert the zoom level to a scaling factor
    const scale = 1 << zoom;

    // the zoom in callback; clip at zoom level 0
    const zoomin = () => (zoom > 0 ? setZoom(zoom-1) : null)
    // the zoom out callback; clip at {maxZoom}
    const zoomout = () => (zoom < maxZoom ? setZoom(zoom+1) : null)

    // pick a signal to show
    const [signal, setSignal] = useState("amplitude")

    // this is the shape of the slc, rounded down to the nearest tile multiple
    // N.B.: this clips the tiles that ride on the right and bottom margin of the data set;
    // they aren't whole tiles, so they either need a dynamically generated margin or more
    // complicated request logic
    const gridShape = [0, 1].map(i => Math.floor(rasterShape[i] / tileShape[i] / scale))
    //  the plot shape rounded down to the nearest tile
    const plotShape = [0, 1].map(i => gridShape[i] * tileShape[i])

    // make a copy of the styling of the bitmap container
    var plotStyle = { ...styles.plot }
    // add size information to the {plot} style sheet
    plotStyle.width = `${plotShape[1]}px`
    plotStyle.height = `${plotShape[0]}px`

    // make a copy of the styling of individual tiles
    var tileStyle = { ...styles.tile }
    // and record the {tile} shape
    tileStyle.width = `${tileShape[1]}px`
    tileStyle.height = `${tileShape[0]}px`

    // tile specification factory
    const assembleTileSpec = (first, ...rest) =>
        rest.reduce(
            (accum, val) => `${accum}x${val}`,
            `${first}`
        )

    // assemble the tile shape spec
    const tileShapeSpec = assembleTileSpec(...tileShape)

    // build the tile uri stem
    const baseURI = `/slc/${slc}/${signal}/${zoom}/${tileShapeSpec}/`

    // given the origin of a tile, build its uri
    const tileURI = (origin) => baseURI + assembleTileSpec(...origin)
    // and its title; disabled, for now
    // const tileTitle = (origin) => `(${origin})+(${tileShapeSpec})`

    // the cartesian product calculator
    const cartesian = (first, ...rest) =>
        rest.reduce(
            (prefix, arg, foo) =>
                prefix.flatMap(prefixEntry =>
                    arg.map(idx => [prefixEntry, tileShape[foo + 1] * idx].flat())
                ),
            // convert {first} into an array of consecutive integers
            first.map(idx => tileShape[0] * idx)
        )
    // build an array of indices
    const indices = [0, 1].map(i => Array(gridShape[i]).fill().map((_, j) => j))
    // use this to build an array of tile origin in SLC coordinates
    const tileOrigins = cartesian(...indices)

    // build the container, fill it, and return it
    return (
        <section style={styles.slc}>
            <div style={styles.viewport}>
                {/* the SLC signal toolbox */}
                <SLC select={setSignal} style={styles.slcToolbox} />
                {/* the zoom toolbox */}
                <Zoom zoomin={zoomin} zoomout={zoomout} style={styles.zoomToolbox} />
                {/* the SLC signal rendering area */}
                <div style={plotStyle} >
                    {tileOrigins.map(origin => {
                         // assemble the tile name
                         const key = tileURI(origin)
                         // the tile uri
                         const uri = tileURI(origin)
                         // render
                         return (
                             <img key={key}
                                  loading="lazy" className="lazyload" data-src={uri}
                                  style={tileStyle}
                             />
                         )
                     })}
                </div>
            </div>
        </section>
    )
}


// publish
export default slc


// end of file
