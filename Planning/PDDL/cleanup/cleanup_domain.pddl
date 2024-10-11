(define (domain cleanup)
    (:requirements :strips :typing)
    (:types 
        gripper environment-object - object
        table tabletop-object - environment-object
        block container - tabletop-object
        mug drawer - container
    ) 
    
    ;; Define predicates
    (:predicates
        (small-enough-to-pick-up ?tabletop-object - tabletop-object) ; whether the object can be picked up.
        (directly-on-table ?tabletop-object - tabletop-object ?table - table) ; whether the object is on the table directly making contact with the table.
        (large-enough-for-gripper-to-reach-inside ?container - container) ; whether the gripper can reach inside the container.
        (occupying-gripper ?tabletop-object - tabletop-object ?gripper - gripper) ; whether the object is occupying the gripper. If true, `free gripper` should be false.
        (in ?tabletop-object - tabletop-object ?container - container) ; whether the object is in the container.
        (open ?container - container) ; whether the container is open.
        (free ?gripper - gripper) ; whether the gripper is not occupied by anything. If true, there should be no true `occupying-gripper` atoms.
    )
    
    ;; Define actions using the predicates given
    (:action pick-up-from-tabletop
        :parameters (?tabletop-object - tabletop-object ?table - table ?gripper - gripper) 
        :precondition (and (directly-on-table ?tabletop-object ?table) (small-enough-to-pick-up ?tabletop-object) (free ?gripper)) 
        :effect (and (occupying-gripper ?tabletop-object ?gripper) (not (directly-on-table ?tabletop-object ?table)) (not (free ?gripper))) 
    )

    (:action free-gripper
        :parameters (?tabletop-object - tabletop-object ?gripper - gripper)
        :precondition (and (not (small-enough-to-pick-up ?tabletop-object)) (occupying-gripper ?tabletop-object ?gripper) (not (free ?gripper)))
        :effect (and (not (occupying-gripper ?tabletop-object ?gripper)) (free ?gripper))
    )
    
    (:action open-drawer
        :parameters (?drawer - drawer ?gripper - gripper)
        :precondition (and (not (open ?drawer)) (free ?gripper))
        :effect (and (open ?drawer) (occupying-gripper ?drawer ?gripper) (not (free ?gripper)))
    )

    (:action close-drawer
        :parameters (?drawer - drawer ?gripper - gripper)
        :precondition (and (open ?drawer) (free ?gripper))
        :effect (and (not (open ?drawer)) (occupying-gripper ?drawer ?gripper) (not (free ?gripper)))
    )

    (:action place-in-drawer-from-gripper
        :parameters (?drawer - drawer ?tabletop-object - tabletop-object ?gripper - gripper)
        :precondition (and (open ?drawer) (large-enough-for-gripper-to-reach-inside ?drawer) (occupying-gripper ?tabletop-object ?gripper) (not (free ?gripper)))
        :effect (and (in ?tabletop-object ?drawer) (not (occupying-gripper ?tabletop-object ?gripper)) (free ?gripper))
    )
    
)