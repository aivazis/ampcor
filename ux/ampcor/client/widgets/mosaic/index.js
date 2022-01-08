// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2022 all rights reserved


// externals
import path from 'path'
import React from 'react'
// locals
import styles from './style'
import { MosaicContext } from '~/context'
import { Tile } from 'widgets'


// currently, a tile is an {img} that's lazy loaded from a {uri}
const widget = React.forwardRef(({ uri, origins, style }, mosaicRef) => {
    // get access to the current pan shift
    const [pan, setPan] = MosaicContext.usePan()

    // schedule a scroll
    React.useEffect(() => {
        // check whether i have an adjustment
        if (!pan) {
            // and if not, do nothing
            return
        }
        // otherwise, get the viewport
        const viewport = mosaicRef.current
        // scroll
        viewport.scroll({
            top: pan.top,
            left: pan.left,
            behavior: "auto",
        })
        // clear the pan
        setPan(null)
    })

    // content styling
    // mix the viewport style
    const viewportStyle = {
        // using the supplied and local styles
        ...styles.viewport, ...style?.viewport,
    }
    // mix the mosaic style
    const mosaicStyle = {
        // using the supplied and local styles
        ...styles.mosaic, ...style?.mosaic,
    }
    // finally, the tile style is a mix of
    const tileStyle = {
        // the supplied styling
        ...styles.tile, ...style?.tile,
    }

    // fold the indices in a tile origin into a string
    const foldRanks = (first, ...rest) => rest.reduce(
        // by looping over the indices and tacking on each one to the partial result
        (partial, index) => `${partial}x${index}`,
        // which is primed by converting the first index into a string
        first.toString()
    )
    // given the origin of a tile, build its uri
    const tileURI = (origin) => path.join(uri, foldRanks(...origin))

    // on double click, center to the mouse coordinates
    const center = ({ clientX, clientY }) => {
        // get the viewport
        const viewport = mosaicRef?.current
        // if it has been rendered
        if (viewport) {
            // get the viewport bounding box
            const box = viewport.getBoundingClientRect()

            // compute the location of the click relative to the viewport
            const x = clientX - box.left
            const y = clientY - box.top
            // get the size of the viewport
            const width = viewport.clientWidth
            const height = viewport.clientHeight
            // get the current scroll position
            const left = viewport.scrollLeft
            const top = viewport.scrollTop
            // and scroll to the new location
            viewport.scroll({
                top: top + y - height / 2,
                left: left + x - width / 2,
                behavior: "auto",
            })
        }
    }

    // render
    return (
        <div ref={mosaicRef} style={viewportStyle} onDoubleClick={center} >
            <div style={mosaicStyle} >
                {origins.map(origin => {
                    // form the tile uri
                    const uri = tileURI(origin)
                    // render
                    return (
                        <Tile key={uri} uri={uri} style={tileStyle} />
                    )
                })}
            </div>
        </div>
    )
})


// publish
export default widget


// end of file
