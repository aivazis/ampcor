// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// get colors
import { wheel, theme } from '~/palette'
// nad base styling
import base from '~/panels/styles'


// tool styling; see '~/widgets/tool/*.js' for details
const tool = {
    // the base state
    base: {
        box: {
            flex: "none",
            width: "2em",
            height: "2em",
            color: theme.banner.name,
            // backgroundColor: theme.page.background,
        },

        group: {
            transform: "scale(0.4)",
        },

        frame: {
            fill: wheel.gray.obsidian,
            fillOpacity: 0.4,
            stroke: wheel.gray.steel,
            strokeOpacity: 0.75,
            strokeWidth: "5px",
        },

        icon: {
            stroke: theme.banner.name,
            strokeOpacity: 0.75,
            strokeWidth: "5px",
        },

    },
    inactive: {
        frame: {
            fillOpacity: .2,
        },
        icon: {
            strokeOpacity: .2,
        },
    },
    selected: {
        frame: {
            fillOpacity: 1.0,
            stroke: theme.banner.name,
        },
    }
}


// publish
export default {

    // the outer container; it owns and positions the toolboxes and must not permit scrolling,
    // otherwise the controls will not be at a fixed location
    panel: {
        // inherit
        ...base.panel,
    },

    // tile rendering
    data: {
        // the viewport; the intermediate layer that absorbs the explicitly sized mosaic
        // and enables scrolling
        viewport: {
        },

        // the bitmap renderer
        mosaic: {
        },

        // individual tiles
        tile: {
            // size hints; the code must override with the actual values
            width: "256px",
            height: "256px",
        },
    },

    // slc tool box styling
    slcToolbox: {
        toolbox: {
            // positioning
            position: "absolute",
            right: "1.0em",
            top: "0.75em",
            // size
            width: "2.0em",
            height: "6.3em", // 3 tools at 2em each, plus some space in between
            // styling
            opacity: "0.75",
            // backgroundColor: theme.page.background,
            // border: `1px solid ${wheel.gray.basalt}`,
        },

        // tool styling; see '~/widgets/tool/*.js' for details
        tool: {
            // start with the default established up top
            ...tool
        },
    },

    // zoom tool box styling
    zoomToolbox: {
        toolbox: {
            // positioning
            position: "absolute",
            right: "1.0em",
            bottom: "0.75em",
            // size
            width: "2.0em",
            height: "4.0em", // 2 tools at 2em each, no space in between
            // styling
            opacity: "0.75",
            // backgroundColor: theme.page.background,
            // border: `1px solid ${wheel.gray.basalt}`,
        },

        // tool styling; see '~/widgets/tool/*.js' for details
        tool: {
            // start with the default established up top
            ...tool
        },
    },

}


// end of file
