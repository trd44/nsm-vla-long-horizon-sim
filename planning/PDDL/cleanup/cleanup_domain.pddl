(define (domain cleanup)
    (:requirements :strips :typing)
    (:types 
        gripper table tabletop-object - object
        block container - tabletop-object
        mug drawer - container
    ) 
    
    ;; Define predicates
    (:predicates
        (closed ?container - container) ; whether the container is completely closed.
        (directly-on-table ?tabletop-object - tabletop-object ?table - table) ; whether the object is on the table directly making contact with the table.
        (exclusively-occupying-gripper ?tabletop-object - tabletop-object ?gripper - gripper) ; whether the object is occupying the gripper. If true, `free gripper` should be false.
        (free ?gripper - gripper) ; whether the gripper is not occupied by anything. If true, there should be no true `exclusively-occupying-gripper` atoms.
        (inside ?tabletop-object - tabletop-object ?container - container) ; whether the object is inside the container.
        (large-enough-for-gripper-to-reach-inside ?container - container ?gripper - gripper) ; whether the gripper can reach inside the container.
        (open-enough-to-fit-through ?container - container ?tabletop-object - tabletop-object) ; whether the container is open enough to fit the object through the opening
        (small-enough-for-gripper-to-pick-up ?tabletop-object - tabletop-object ?gripper - gripper) ; whether the object can be picked up.
        
    )
    
    ;; Define actions using the predicates given
    (:action pick-up-from-tabletop
        :parameters (?tabletop-object - tabletop-object ?table - table ?gripper - gripper) 
        :precondition (and (directly-on-table ?tabletop-object ?table) (small-enough-for-gripper-to-pick-up ?tabletop-object ?gripper) (free ?gripper)) 
        :effect (and (exclusively-occupying-gripper ?tabletop-object ?gripper) (not (directly-on-table ?tabletop-object ?table)) (not (free ?gripper))) 
    )

    (:action free-gripper-from-large-object
        :parameters (?tabletop-object - tabletop-object ?gripper - gripper)
        :precondition (and (not (small-enough-for-gripper-to-pick-up ?tabletop-object ?gripper)) (exclusively-occupying-gripper ?tabletop-object ?gripper) (not (free ?gripper)))
        :effect (and (not (exclusively-occupying-gripper ?tabletop-object ?gripper)) (free ?gripper))
    )
    
    (:action open-drawer-for-object
        :parameters (?drawer - drawer ?tabletop-object - tabletop-object ?gripper - gripper)
        :precondition (and (not (open-enough-to-fit-through ?drawer ?tabletop-object)) (free ?gripper))
        :effect (and (open-enough-to-fit-through ?drawer ?tabletop-object) (exclusively-occupying-gripper ?drawer ?gripper) (not (free ?gripper)))
    )

    (:action close-drawer
        :parameters (?drawer - drawer ?gripper - gripper)
        :precondition (and (not (closed ?drawer)) (free ?gripper))
        :effect (and (closed ?drawer) (exclusively-occupying-gripper ?drawer ?gripper) (not (free ?gripper)))
    )

    (:action place-in-drawer-from-gripper
        :parameters (?drawer - drawer ?tabletop-object - tabletop-object ?gripper - gripper)
        :precondition (and (open-enough-to-fit-through ?drawer ?tabletop-object) (large-enough-for-gripper-to-reach-inside ?drawer ?gripper) (exclusively-occupying-gripper ?tabletop-object ?gripper) (not (free ?gripper)))
        :effect (and (inside ?tabletop-object ?drawer) (not (exclusively-occupying-gripper ?tabletop-object ?gripper)) (free ?gripper))
    )
    
)