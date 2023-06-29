import { Place } from "@mui/icons-material";
import { getLocationTypeById } from "../../constants/LocationTypes";
import { Location } from "../../types/Location";
import { Position } from "../../types/Position";

interface LocationRendererProps {
    locations: Location[];
    resolution: number;
    origin: Position;
    width: number;
    height: number;
    pixelWidth: number;
    pixelHeight: number;
};

export function LocationRenderer(props: LocationRendererProps) {
    return (
        <div
            style={{
                position: "absolute",
                width: `${props.pixelWidth}px`,
                height: `${props.pixelHeight}px`,
            }}
        >
            {props.locations.map(location => <div
                style={{
                    color: getLocationTypeById(location.type).color,
                    zIndex: 1,
                    position: "absolute",
                    left: `${location.position.x / props.width * props.pixelWidth - 17.5}px`,
                    top: `${location.position.y / props.height * props.pixelHeight - 17.5}px`,
                }}>
                <Place fontSize="large"/>
            </div>)}
        </div>
    );

}