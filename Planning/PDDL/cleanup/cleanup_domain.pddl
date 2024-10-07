(define (domain cleanup)
    (:requirements :strips :typing)
    (:types 
        mug block drawer table gripper - object
    ) 
    
    (:predicates
        (can-hold ?obj - object)
        (can-open ?obj - object)
        (can-contain ?container - object ?obj - object)
        (on-table ?obj - object ?table - table)
        (holding ?obj - object)
        (in ?obj - object ?container - object)
        (open ?container - object)
        (free ?gripper - gripper) 
    )
    
    (:action pick-up-tabletop
        :parameters (?obj - object ?table - table ?gripper - gripper) 
        :precondition (and (on-table ?obj ?table) (can-hold ?obj) (free ?gripper)) 
        :effect (and (holding ?obj) (not (on-table ?obj ?table)) (not (free ?gripper))) 
    )
    
    (:action open-drawer
        :parameters (?drawer - drawer ?gripper - gripper)
        :precondition (and (can-open ?drawer) (not (open ?drawer)) (free ?gripper))
        :effect (open ?drawer)
    )

    (:action close-drawer
        :parameters (?drawer - drawer ?gripper - gripper)
        :precondition (and (open ?drawer) (free ?gripper))
        :effect (not (open ?drawer))
    )

    (:action put-in-drawer
        :parameters (?drawer - drawer ?obj - object ?gripper - gripper)
        :precondition (and (open ?drawer) (can-contain ?drawer ?obj) (holding ?obj) (not (free ?gripper)))
        :effect (and (in ?obj ?drawer) (not (holding ?obj)) (free ?gripper))
    )
    
)