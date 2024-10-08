(define (domain nut_assembly)
    (:requirements :strips :typing)
    (:types 
        gripper table tabletop-object - object
        nut peg - tabletop-object
        round-peg square-peg - peg
        round-nut square-nut - nut
    ) 
    
    ;; Define predicates
    (:predicates
        (can-pick-up ?tabletop-object - tabletop-object) ; whether the nut can be picked up.
        (on-table ?tabletop-object - tabletop-object ?table - table) ; whether the object is on the table directly making contact with the table.
        (on ?nut - nut ?peg - peg) ; whether the nut is on the peg.
        (occupying-gripper ?tabletop-object - tabletop-object ?gripper - gripper) ; whether the object is occupying the gripper. If true, `free gripper` should be false.
        (match ?nut - nut ?peg - peg) ; whether the nut matches the peg.
        (free ?gripper - gripper) ; whether the gripper is not occupied by anything. If true, there should be no true `occupying-gripper` atoms.
    )
    
    ;; Define actions using the predicates given
    (:action pick-up-nut-from-tabletop
        :parameters (?nut - nut ?table - table ?gripper - gripper) 
        :precondition (and (on-table ?nut ?table) (can-pick-up ?nut) (free ?gripper)) 
        :effect (and (occupying-gripper ?nut ?gripper) (not (on-table ?nut ?table)) (not (free ?gripper))) 
    )

    (:action free-gripper
        :parameters (?tabletop-object - tabletop-object ?gripper - gripper)
        :precondition (and (not (can-pick-up ?tabletop-object)) (occupying-gripper ?tabletop-object ?gripper) (not (free ?gripper)))
        :effect (and (not (occupying-gripper ?tabletop-object ?gripper)) (free ?gripper))
    )
    

    (:action put-nut-on-peg
        :parameters (?nut - nut ?peg - peg ?gripper - gripper)
        :precondition (and (match ?nut ?peg) (occupying-gripper ?nut ?gripper) (not (free ?gripper)))
        :effect (and (on ?nut ?peg) (not (occupying-gripper ?nut ?gripper)) (free ?gripper))
    )
    
)