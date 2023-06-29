export interface MapDescriptor {
    image: string;
    resolution: number;
    origin: [number, number, number];
    negate: number;
    
    occupied_thresh: number;
    free_thresh: number;

    base_size: [6, 6];
    safety_margin: number;

    distance_threshold: number;

    min_contour_area: number;
    min_roadmap_area: number;
    max_corridor_width: number;
    max_distance_to_connect_points: number;
}